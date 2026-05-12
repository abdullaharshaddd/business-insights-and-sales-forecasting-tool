import prisma from '../../config/database';
import { AddStockInput, RemoveStockInput, AdjustStockInput } from './inventory.schema';
import { NotFoundError, InsufficientStockError, ValidationError } from '../../shared/errors/AppError';
import { MovementType, Prisma } from '@prisma/client';
import { parsePagination } from '../../shared/utils/response';

// ─── Unified Inventory View ─────────────────────────────────────────────────

type UnifiedInventoryItem = {
  source: 'retail' | 'olist' | 'operational';
  productId: string;
  sku: string | null;
  stockcode: string | null;
  name: string;
  description: string | null;
  quantity: number | null;
  basePrice: number | null;
  latestUnitPrice: number | null;
  category: string | null;
  supplier: string | null;
  lastRestockAt: Date | null;
  lastSoldAt: Date | null;
  updatedAt: Date | null;
};

export class InventoryService {
  // ─── Complete Inventory View ─────────────────────────────────────────────
  async getCompleteInventory(query: Record<string, any> = {}) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const search = query.search as string | undefined;
    const source = query.source as string | undefined;

    // Get operational inventory products
    const operationalProducts = await prisma.product.findMany({
      where: search
        ? {
            OR: [
              { sku: { contains: search, mode: 'insensitive' } },
              { name: { contains: search, mode: 'insensitive' } },
              { description: { contains: search, mode: 'insensitive' } },
            ],
          }
        : undefined,
      include: {
        inventory: true,
        category: { select: { name: true } },
        supplier: { select: { name: true } },
      },
    });

    // Get Online Retail products (without inventory tracking)
    let retailProducts: any[] = [];
    if (!source || source === 'retail' || source === 'all') {
      retailProducts = await prisma.retailProduct.findMany({
        where: search
          ? {
              OR: [
                { stockcode: { contains: search, mode: 'insensitive' } },
                { description: { contains: search, mode: 'insensitive' } },
              ],
            }
          : undefined,
        orderBy: { stockcode: 'asc' },
      });
    }

    // Get Olist products (without inventory tracking)
    let olistProducts: any[] = [];
    if (!source || source === 'olist' || source === 'all') {
      olistProducts = await prisma.olistProduct.findMany({
        where: search
          ? { categoryNamePt: { contains: search, mode: 'insensitive' } }
          : undefined,
        orderBy: { productId: 'asc' },
      });
    }

    // Transform to unified format
    const items: UnifiedInventoryItem[] = [];

    // Operational products
    for (const p of operationalProducts) {
      if (source && source !== 'operational' && source !== 'all') continue;
      items.push({
        source: 'operational',
        productId: p.id,
        sku: p.sku,
        stockcode: null,
        name: p.name,
        description: p.description,
        quantity: p.inventory?.quantity ?? null,
        basePrice: p.basePrice ? Number(p.basePrice) : null,
        latestUnitPrice: null,
        category: p.category?.name ?? null,
        supplier: p.supplier?.name ?? null,
        lastRestockAt: p.inventory?.lastRestockAt ?? null,
        lastSoldAt: p.inventory?.lastSoldAt ?? null,
        updatedAt: p.inventory?.updatedAt ?? null,
      });
    }

    // Retail products
    for (const p of retailProducts) {
      items.push({
        source: 'retail',
        productId: p.stockcode,
        sku: null,
        stockcode: p.stockcode,
        name: p.description || p.stockcode,
        description: p.description,
        quantity: null,
        basePrice: null,
        latestUnitPrice: p.latestUnitPrice ? Number(p.latestUnitPrice) : null,
        category: null,
        supplier: null,
        lastRestockAt: null,
        lastSoldAt: null,
        updatedAt: null,
      });
    }

    // Olist products
    for (const p of olistProducts) {
      items.push({
        source: 'olist',
        productId: p.productId,
        sku: null,
        stockcode: null,
        name: p.categoryNamePt || p.productId,
        description: null,
        quantity: null,
        basePrice: null,
        latestUnitPrice: null,
        category: p.categoryNamePt,
        supplier: null,
        lastRestockAt: null,
        lastSoldAt: null,
        updatedAt: null,
      });
    }

    // Sort by source then by name
    items.sort((a, b) => {
      if (a.source !== b.source) {
        const order = { operational: 0, retail: 1, olist: 2 };
        return order[a.source] - order[b.source];
      }
      return a.name.localeCompare(b.name);
    });

    const total = items.length;
    const paginatedItems = items.slice(skip, skip + limit);

    return { items: paginatedItems, total, page, limit };
  }

  // ─── Operational Inventory List ──────────────────────────────────────────
  async findAll(query: Record<string, any>) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const sortBy = query.sortBy || 'updatedAt';
    const order = query.order || 'desc';

    const [items, total] = await Promise.all([
      prisma.inventory.findMany({
        include: {
          product: {
            select: {
              id: true, sku: true, name: true, status: true,
              basePrice: true, reorderPoint: true,
              category: { select: { id: true, name: true } },
              supplier: { select: { id: true, name: true } },
            },
          },
        },
        orderBy: { [sortBy]: order },
        skip,
        take: limit,
      }),
      prisma.inventory.count(),
    ]);

    return { items, total, page, limit };
  }

  // ─── Get inventory for a specific product ───────────────────────────────
  async findByProductId(productId: string) {
    const inventory = await prisma.inventory.findUnique({
      where: { productId },
      include: {
        product: {
          include: {
            category: { select: { id: true, name: true } },
            supplier: { select: { id: true, name: true } },
          },
        },
      },
    });
    if (!inventory) throw new NotFoundError('Inventory for product', productId);
    return inventory;
  }

  // ─── ADD STOCK (transactional) ──────────────────────────────────────────
  async addStock(input: AddStockInput, userId: string) {
    return prisma.$transaction(async (tx) => {
      const inventory = await tx.inventory.findUnique({
        where: { productId: input.productId },
      });
      if (!inventory) throw new NotFoundError('Inventory for product', input.productId);

      const quantityBefore = inventory.quantity;
      const quantityAfter = quantityBefore + input.quantity;

      await tx.inventory.update({
        where: { productId: input.productId },
        data: {
          quantity: quantityAfter,
          lastRestockAt: new Date(),
        },
      });

      await tx.stockMovement.create({
        data: {
          productId: input.productId,
          inventoryId: inventory.id,
          movementType: MovementType.IN,
          quantity: input.quantity,
          quantityBefore,
          quantityAfter,
          reason: input.reason,
          referenceType: input.referenceType || 'manual',
          referenceId: input.referenceId || null,
          performedBy: userId,
        },
      });

      return {
        productId: input.productId,
        quantityBefore,
        quantityAdded: input.quantity,
        quantityAfter,
        movement: 'IN',
      };
    });
  }

  // ─── REMOVE STOCK (transactional with safety check) ─────────────────────
  async removeStock(input: RemoveStockInput, userId: string) {
    return prisma.$transaction(async (tx) => {
      const inventory = await tx.inventory.findUnique({
        where: { productId: input.productId },
      });
      if (!inventory) throw new NotFoundError('Inventory for product', input.productId);

      const available = inventory.quantity - inventory.reservedQty;
      if (available < input.quantity) {
        throw new InsufficientStockError(available, input.quantity);
      }

      const quantityBefore = inventory.quantity;
      const quantityAfter = quantityBefore - input.quantity;

      await tx.inventory.update({
        where: { productId: input.productId },
        data: {
          quantity: quantityAfter,
          lastSoldAt: new Date(),
        },
      });

      await tx.stockMovement.create({
        data: {
          productId: input.productId,
          inventoryId: inventory.id,
          movementType: MovementType.OUT,
          quantity: -input.quantity,
          quantityBefore,
          quantityAfter,
          reason: input.reason,
          referenceType: input.referenceType || 'manual',
          referenceId: input.referenceId || null,
          performedBy: userId,
        },
      });

      return {
        productId: input.productId,
        quantityBefore,
        quantityRemoved: input.quantity,
        quantityAfter,
        movement: 'OUT',
      };
    });
  }

  // ─── ADJUST STOCK (correction — transactional) ──────────────────────────
  async adjustStock(input: AdjustStockInput, userId: string) {
    return prisma.$transaction(async (tx) => {
      const inventory = await tx.inventory.findUnique({
        where: { productId: input.productId },
      });
      if (!inventory) throw new NotFoundError('Inventory for product', input.productId);

      const quantityBefore = inventory.quantity;
      const quantityAfter = input.newQuantity;
      const diff = quantityAfter - quantityBefore;

      if (diff === 0) {
        throw new ValidationError('New quantity is the same as current quantity');
      }

      await tx.inventory.update({
        where: { productId: input.productId },
        data: { quantity: quantityAfter },
      });

      await tx.stockMovement.create({
        data: {
          productId: input.productId,
          inventoryId: inventory.id,
          movementType: MovementType.ADJUSTMENT,
          quantity: diff,
          quantityBefore,
          quantityAfter,
          reason: input.reason,
          referenceType: 'adjustment',
          performedBy: userId,
        },
      });

      return {
        productId: input.productId,
        quantityBefore,
        quantityAfter,
        adjustment: diff,
        movement: 'ADJUSTMENT',
      };
    });
  }

  // ─── Stock Movement History ─────────────────────────────────────────────
  async getMovements(productId: string, query: Record<string, any>) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const where: Prisma.StockMovementWhereInput = { productId };

    if (query.type) {
      where.movementType = query.type as MovementType;
    }

    const [movements, total] = await Promise.all([
      prisma.stockMovement.findMany({
        where,
        include: {
          user: { select: { id: true, fullName: true, email: true } },
        },
        orderBy: { createdAt: 'desc' },
        skip,
        take: limit,
      }),
      prisma.stockMovement.count({ where }),
    ]);

    return { movements, total, page, limit };
  }

  // ─── Low Stock Alerts ───────────────────────────────────────────────────
  async getLowStockAlerts() {
    const alerts = await prisma.$queryRaw<Array<{
      product_id: string;
      sku: string;
      name: string;
      quantity: number;
      reserved_qty: number;
      reorder_point: number;
      reorder_qty: number;
      supplier_name: string | null;
      category_name: string | null;
    }>>`
      SELECT
        p.id as product_id, p.sku, p.name,
        i.quantity, i.reserved_qty,
        p.reorder_point, p.reorder_qty,
        s.name as supplier_name,
        c.name as category_name
      FROM products p
      JOIN inventory i ON i.product_id = p.id
      LEFT JOIN suppliers s ON p.supplier_id = s.id
      LEFT JOIN categories c ON p.category_id = c.id
      WHERE p.status = 'active' AND i.quantity <= p.reorder_point
      ORDER BY (i.quantity - p.reorder_point) ASC
    `;

    return alerts.map((a) => ({
      productId: a.product_id,
      quantity: a.quantity,
      product: {
        name: a.name,
        sku: a.sku,
        reorderPoint: a.reorder_point
      },
      severity: a.quantity === 0 ? 'critical' : a.quantity <= Math.floor(a.reorder_point / 2) ? 'high' : 'medium',
    }));
  }

  // ─── Inventory Stats ────────────────────────────────────────────────────
  async getStats() {
    const [operationalCount, retailCount, olistCount, lowStockCount] = await Promise.all([
      prisma.product.count(),
      prisma.retailProduct.count(),
      prisma.olistProduct.count(),
      prisma.$queryRaw<Array<{ count: BigInt }>>`
        SELECT COUNT(*) as count FROM products p
        JOIN inventory i ON i.product_id = p.id
        WHERE p.status = 'active' AND i.quantity <= p.reorder_point
      `,
    ]);

    return {
      operational: operationalCount,
      retail: retailCount,
      olist: olistCount,
      total: operationalCount + retailCount + olistCount,
      lowStockAlerts: Number(lowStockCount[0]?.count || 0),
    };
  }
}

export const inventoryService = new InventoryService();