import prisma from '../../config/database';
import { CreateSupplierInput, UpdateSupplierInput } from './supplier.schema';
import { NotFoundError } from '../../shared/errors/AppError';
import { parsePagination } from '../../shared/utils/response';
import { Prisma } from '@prisma/client';

export class SupplierService {
  async findAll(query: Record<string, any>) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const where: Prisma.SupplierWhereInput = { isActive: true };

    if (query.search) {
      where.OR = [
        { name: { contains: query.search, mode: 'insensitive' } },
        { city: { contains: query.search, mode: 'insensitive' } },
        { contactPerson: { contains: query.search, mode: 'insensitive' } },
      ];
    }

    const [suppliers, total] = await Promise.all([
      prisma.supplier.findMany({
        where,
        include: { _count: { select: { products: true, purchaseOrders: true } } },
        orderBy: { name: 'asc' },
        skip,
        take: limit,
      }),
      prisma.supplier.count({ where }),
    ]);

    return { suppliers, total, page, limit };
  }

  async findById(id: string) {
    const supplier = await prisma.supplier.findUnique({
      where: { id },
      include: {
        products: {
          select: { id: true, sku: true, name: true, status: true, basePrice: true },
          where: { status: 'active' },
        },
        _count: { select: { products: true, purchaseOrders: true } },
      },
    });
    if (!supplier) throw new NotFoundError('Supplier', id);
    return supplier;
  }

  async create(input: CreateSupplierInput) {
    return prisma.supplier.create({
      data: input,
      include: { _count: { select: { products: true } } },
    });
  }

  async update(id: string, input: UpdateSupplierInput) {
    await this.findById(id);
    return prisma.supplier.update({
      where: { id },
      data: input,
      include: { _count: { select: { products: true } } },
    });
  }

  async deactivate(id: string) {
    await this.findById(id);
    return prisma.supplier.update({
      where: { id },
      data: { isActive: false },
    });
  }
}

export const supplierService = new SupplierService();
