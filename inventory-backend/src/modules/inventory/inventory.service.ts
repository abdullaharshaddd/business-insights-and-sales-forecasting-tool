import prisma from '../../config/database';
import { AddStockInput, RemoveStockInput, AdjustStockInput } from './inventory.schema';
import { NotFoundError, InsufficientStockError, ValidationError } from '../../shared/errors/AppError';
import { MovementType, Prisma } from '@prisma/client';
import { parsePagination } from '../../shared/utils/response';

export class InventoryService {
  // ─── List all inventory with product details ────────────────────────────
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
      // Lock the inventory row
      const inventory = await tx.inventory.findUnique({
        where: { productId: input.productId },
      });
      if (!inventory) throw new NotFoundError('Inventory for product', input.productId);

      const quantityBefore = inventory.quantity;
      const quantityAfter = quantityBefore + input.quantity;

      // Update inventory
      const updated = await tx.inventory.update({
        where: { productId: input.productId },
        data: {
          quantity: quantityAfter,
          lastRestockAt: new Date(),
        },
      });

      // Record movement
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

      const updated = await tx.inventory.update({
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
}

export const inventoryService = new InventoryService();
