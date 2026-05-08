import prisma from '../../config/database';
import { CreateProductInput, UpdateProductInput } from './product.schema';
import { NotFoundError, ConflictError } from '../../shared/errors/AppError';
import { Prisma, ProductStatus } from '@prisma/client';
import { parsePagination } from '../../shared/utils/response';

interface ProductQuery {
  page?: string;
  limit?: string;
  search?: string;
  categoryId?: string;
  supplierId?: string;
  status?: ProductStatus;
  sortBy?: string;
  order?: 'asc' | 'desc';
}

export class ProductService {
  async findAll(query: ProductQuery) {
    const { page, limit } = parsePagination(query as any);
    const skip = (page - 1) * limit;

    const where: Prisma.ProductWhereInput = {};

    if (query.search) {
      where.OR = [
        { name: { contains: query.search, mode: 'insensitive' } },
        { sku: { contains: query.search, mode: 'insensitive' } },
        { description: { contains: query.search, mode: 'insensitive' } },
      ];
    }
    if (query.categoryId) where.categoryId = query.categoryId;
    if (query.supplierId) where.supplierId = query.supplierId;
    if (query.status) where.status = query.status;

    const orderBy: Prisma.ProductOrderByWithRelationInput = {
      [query.sortBy || 'createdAt']: query.order || 'desc',
    };

    const [products, total] = await Promise.all([
      prisma.product.findMany({
        where,
        include: {
          category: { select: { id: true, name: true, slug: true } },
          supplier: { select: { id: true, name: true } },
          inventory: { select: { quantity: true, reservedQty: true } },
        },
        orderBy,
        skip,
        take: limit,
      }),
      prisma.product.count({ where }),
    ]);

    return { products, total, page, limit };
  }

  async findById(id: string) {
    const product = await prisma.product.findUnique({
      where: { id },
      include: {
        category: true,
        supplier: true,
        inventory: true,
      },
    });
    if (!product) throw new NotFoundError('Product', id);
    return product;
  }

  async create(input: CreateProductInput) {
    // Check SKU uniqueness
    const existingSku = await prisma.product.findUnique({ where: { sku: input.sku } });
    if (existingSku) throw new ConflictError(`Product with SKU '${input.sku}' already exists`);

    if (input.barcode) {
      const existingBarcode = await prisma.product.findUnique({ where: { barcode: input.barcode } });
      if (existingBarcode) throw new ConflictError(`Product with barcode '${input.barcode}' already exists`);
    }

    // Validate category exists
    if (input.categoryId) {
      const cat = await prisma.category.findUnique({ where: { id: input.categoryId } });
      if (!cat) throw new NotFoundError('Category', input.categoryId);
    }

    // Validate supplier exists
    if (input.supplierId) {
      const sup = await prisma.supplier.findUnique({ where: { id: input.supplierId } });
      if (!sup) throw new NotFoundError('Supplier', input.supplierId);
    }

    // Create product and its inventory record in a transaction
    const product = await prisma.$transaction(async (tx) => {
      const created = await tx.product.create({
        data: input as any,
        include: { category: true, supplier: true },
      });

      // Auto-create inventory record
      await tx.inventory.create({
        data: { productId: created.id, quantity: 0, reservedQty: 0 },
      });

      return created;
    });

    return product;
  }

  async update(id: string, input: UpdateProductInput) {
    await this.findById(id);

    if (input.sku) {
      const existing = await prisma.product.findFirst({ where: { sku: input.sku, NOT: { id } } });
      if (existing) throw new ConflictError(`Product with SKU '${input.sku}' already exists`);
    }

    if (input.barcode) {
      const existing = await prisma.product.findFirst({ where: { barcode: input.barcode, NOT: { id } } });
      if (existing) throw new ConflictError(`Product with barcode '${input.barcode}' already exists`);
    }

    return prisma.product.update({
      where: { id },
      data: input as any,
      include: { category: true, supplier: true, inventory: true },
    });
  }

  async updateStatus(id: string, status: ProductStatus) {
    await this.findById(id);
    return prisma.product.update({
      where: { id },
      data: { status },
      include: { category: true, inventory: true },
    });
  }

  async findLowStock() {
    const products = await prisma.product.findMany({
      where: {
        status: 'active',
        inventory: {
          quantity: { lte: prisma.product.fields.reorderPoint as any },
        },
      },
      include: {
        inventory: true,
        category: { select: { id: true, name: true } },
        supplier: { select: { id: true, name: true } },
      },
    });

    // Fallback: raw query for the computed comparison
    const lowStockProducts = await prisma.$queryRaw<Array<{ id: string }>>`
      SELECT p.id FROM products p
      JOIN inventory i ON i.product_id = p.id
      WHERE p.status = 'active' AND i.quantity <= p.reorder_point
    `;

    const ids = lowStockProducts.map((p) => p.id);
    if (ids.length === 0) return [];

    return prisma.product.findMany({
      where: { id: { in: ids } },
      include: {
        inventory: true,
        category: { select: { id: true, name: true } },
        supplier: { select: { id: true, name: true } },
      },
      orderBy: { name: 'asc' },
    });
  }
}

export const productService = new ProductService();
