import prisma from '../../config/database';
import { CreatePOInput, ReceivePOInput } from './purchase-order.schema';
import { NotFoundError, ValidationError } from '../../shared/errors/AppError';
import { PurchaseOrderStatus, MovementType, Prisma } from '@prisma/client';
import { parsePagination } from '../../shared/utils/response';

export class PurchaseOrderService {
  // ─── Generate PO number ─────────────────────────────────────────────────
  private async generatePONumber(): Promise<string> {
    const year = new Date().getFullYear();
    const count = await prisma.purchaseOrder.count({
      where: {
        createdAt: {
          gte: new Date(`${year}-01-01`),
          lt: new Date(`${year + 1}-01-01`),
        },
      },
    });
    return `PO-${year}-${String(count + 1).padStart(5, '0')}`;
  }

  // ─── List POs ───────────────────────────────────────────────────────────
  async findAll(query: Record<string, any>) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const where: Prisma.PurchaseOrderWhereInput = {};
    if (query.status) where.status = query.status as PurchaseOrderStatus;
    if (query.supplierId) where.supplierId = query.supplierId;

    const [orders, total] = await Promise.all([
      prisma.purchaseOrder.findMany({
        where,
        include: {
          supplier: { select: { id: true, name: true } },
          user: { select: { id: true, fullName: true } },
          _count: { select: { items: true } },
        },
        orderBy: { createdAt: 'desc' },
        skip,
        take: limit,
      }),
      prisma.purchaseOrder.count({ where }),
    ]);

    return { orders, total, page, limit };
  }

  // ─── Get PO detail ─────────────────────────────────────────────────────
  async findById(id: string) {
    const po = await prisma.purchaseOrder.findUnique({
      where: { id },
      include: {
        supplier: true,
        user: { select: { id: true, fullName: true, email: true } },
        items: {
          include: {
            product: { select: { id: true, sku: true, name: true } },
          },
        },
      },
    });
    if (!po) throw new NotFoundError('Purchase Order', id);
    return po;
  }

  // ─── Create PO ──────────────────────────────────────────────────────────
  async create(input: CreatePOInput, userId: string) {
    // Validate supplier
    const supplier = await prisma.supplier.findUnique({ where: { id: input.supplierId } });
    if (!supplier) throw new NotFoundError('Supplier', input.supplierId);

    // Validate all products exist
    const productIds = input.items.map((i) => i.productId);
    const products = await prisma.product.findMany({ where: { id: { in: productIds } } });
    if (products.length !== productIds.length) {
      throw new ValidationError('One or more product IDs are invalid');
    }

    const poNumber = await this.generatePONumber();
    const totalAmount = input.items.reduce(
      (sum, item) => sum + item.quantityOrdered * item.unitCost, 0
    );

    return prisma.purchaseOrder.create({
      data: {
        poNumber,
        supplierId: input.supplierId,
        status: 'draft',
        expectedDate: input.expectedDate ? new Date(input.expectedDate) : null,
        totalAmount,
        notes: input.notes,
        createdBy: userId,
        items: {
          create: input.items.map((item) => ({
            productId: item.productId,
            quantityOrdered: item.quantityOrdered,
            unitCost: item.unitCost,
          })),
        },
      },
      include: {
        supplier: { select: { id: true, name: true } },
        items: {
          include: { product: { select: { id: true, sku: true, name: true } } },
        },
      },
    });
  }

  // ─── Update PO Status ──────────────────────────────────────────────────
  async updateStatus(id: string, status: 'submitted' | 'confirmed' | 'cancelled') {
    const po = await this.findById(id);

    // Status transition validation
    const validTransitions: Record<string, string[]> = {
      draft: ['submitted', 'cancelled'],
      submitted: ['confirmed', 'cancelled'],
      confirmed: ['cancelled'],
      partial: ['cancelled'],
    };

    const allowed = validTransitions[po.status] || [];
    if (!allowed.includes(status)) {
      throw new ValidationError(
        `Cannot transition from '${po.status}' to '${status}'. Allowed: ${allowed.join(', ')}`
      );
    }

    return prisma.purchaseOrder.update({
      where: { id },
      data: { status },
      include: {
        supplier: { select: { id: true, name: true } },
        items: { include: { product: { select: { id: true, sku: true, name: true } } } },
      },
    });
  }

  // ─── Receive PO (updates inventory — TRANSACTIONAL) ─────────────────────
  async receive(id: string, input: ReceivePOInput, userId: string) {
    const po = await this.findById(id);

    if (!['confirmed', 'partial'].includes(po.status)) {
      throw new ValidationError(`Cannot receive PO with status '${po.status}'. Must be 'confirmed' or 'partial'.`);
    }

    return prisma.$transaction(async (tx) => {
      let allFullyReceived = true;

      for (const receivedItem of input.items) {
        const poItem = po.items.find((i) => i.id === receivedItem.purchaseOrderItemId);
        if (!poItem) {
          throw new NotFoundError('Purchase Order Item', receivedItem.purchaseOrderItemId);
        }

        const totalReceived = poItem.quantityReceived + receivedItem.quantityReceived;
        if (totalReceived > poItem.quantityOrdered) {
          throw new ValidationError(
            `Cannot receive ${receivedItem.quantityReceived} units for ${(poItem as any).product.name}. ` +
            `Already received ${poItem.quantityReceived} of ${poItem.quantityOrdered} ordered.`
          );
        }

        // Update PO item
        await tx.purchaseOrderItem.update({
          where: { id: poItem.id },
          data: { quantityReceived: totalReceived },
        });

        if (totalReceived < poItem.quantityOrdered) {
          allFullyReceived = false;
        }

        // Update inventory if quantity received > 0
        if (receivedItem.quantityReceived > 0) {
          const inventory = await tx.inventory.findUnique({
            where: { productId: poItem.productId },
          });

          if (inventory) {
            const qtyBefore = inventory.quantity;
            const qtyAfter = qtyBefore + receivedItem.quantityReceived;

            await tx.inventory.update({
              where: { productId: poItem.productId },
              data: { quantity: qtyAfter, lastRestockAt: new Date() },
            });

            await tx.stockMovement.create({
              data: {
                productId: poItem.productId,
                inventoryId: inventory.id,
                movementType: MovementType.IN,
                quantity: receivedItem.quantityReceived,
                quantityBefore: qtyBefore,
                quantityAfter: qtyAfter,
                reason: `Received from PO ${po.poNumber}`,
                referenceType: 'purchase_order',
                referenceId: po.id,
                performedBy: userId,
              },
            });
          }
        }
      }

      // Update PO status
      const newStatus = allFullyReceived ? 'received' : 'partial';
      await tx.purchaseOrder.update({
        where: { id },
        data: {
          status: newStatus as PurchaseOrderStatus,
          ...(allFullyReceived && { receivedDate: new Date() }),
        },
      });

      return { poNumber: po.poNumber, status: newStatus, itemsProcessed: input.items.length };
    });
  }
}

export const purchaseOrderService = new PurchaseOrderService();
