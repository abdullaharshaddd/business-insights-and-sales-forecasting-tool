import prisma from '../../config/database';
import { CreateSaleInput, BatchCreateSalesInput, SaleItemInput } from './sales.schema';
import { NotFoundError, ValidationError } from '../../shared/errors/AppError';
import { SaleStatus, MovementType, Prisma } from '@prisma/client';
import { parsePagination } from '../../shared/utils/response';

export class SalesService {
  // ─── Generate Sale number ─────────────────────────────────────────────────
  private async generateSaleNumber(tx?: Prisma.TransactionClient): Promise<string> {
    const db = tx || prisma;
    const year = new Date().getFullYear();
    const count = await db.sale.count({
      where: {
        createdAt: {
          gte: new Date(`${year}-01-01`),
          lt: new Date(`${year + 1}-01-01`),
        },
      },
    });
    return `SAL-${year}-${String(count + 1).padStart(5, '0')}`;
  }

  // ─── Metrics Recalculation Engine ─────────────────────────────────────────
  async recalculateDailyMetrics(date: Date, tx?: Prisma.TransactionClient) {
    const db = tx || prisma;
    const startOfDay = new Date(date);
    startOfDay.setUTCHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setUTCHours(23, 59, 59, 999);

    const sales = await db.sale.findMany({
      where: {
        status: 'completed',
        saleDate: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
    });

    const totalSales = sales.length;
    let revenue = 0;
    let cost = 0;
    let profit = 0;

    for (const sale of sales) {
      revenue += Number(sale.totalAmount);
      cost += Number(sale.totalCost);
      profit += Number(sale.totalProfit);
    }

    const marginPct = revenue > 0 ? (profit / revenue) * 100 : 0;

    await db.salesMetrics.upsert({
      where: { date: startOfDay },
      update: {
        totalSales,
        revenue,
        cost,
        profit,
        marginPct,
      },
      create: {
        date: startOfDay,
        totalSales,
        revenue,
        cost,
        profit,
        marginPct,
      },
    });
  }

  // ─── List Sales ───────────────────────────────────────────────────────────
  async findAll(query: Record<string, any>) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const where: Prisma.SaleWhereInput = {};
    if (query.status) where.status = query.status as SaleStatus;
    if (query.customerId) where.customerId = query.customerId;
    if (query.startDate && query.endDate) {
      where.saleDate = {
        gte: new Date(query.startDate as string),
        lte: new Date(query.endDate as string),
      };
    }

    const [sales, total] = await Promise.all([
      prisma.sale.findMany({
        where,
        include: {
          user: { select: { id: true, fullName: true } },
          _count: { select: { items: true } },
        },
        orderBy: { createdAt: 'desc' },
        skip,
        take: limit,
      }),
      prisma.sale.count({ where }),
    ]);

    return { sales, total, page, limit };
  }

  // ─── Get Sale detail ─────────────────────────────────────────────────────
  async findById(id: string) {
    const sale = await prisma.sale.findUnique({
      where: { id },
      include: {
        user: { select: { id: true, fullName: true, email: true } },
        items: {
          include: {
            product: { select: { id: true, sku: true, name: true } },
          },
        },
      },
    });
    if (!sale) throw new NotFoundError('Sale', id);
    return sale;
  }

  // ─── Create Single Sale (Draft by default) ──────────────────────────────
  async create(input: CreateSaleInput, userId: string) {
    return prisma.$transaction(async (tx) => {
      const productIds = input.items.map((i) => i.productId);
      const products = await tx.product.findMany({ where: { id: { in: productIds } } });
      
      if (products.length !== productIds.length) {
        throw new ValidationError('One or more product IDs are invalid');
      }

      const productMap = new Map(products.map(p => [p.id, p]));

      let totalAmount = 0;
      let totalCost = 0;
      let totalProfit = 0;

      const saleItems = input.items.map((item) => {
        const product = productMap.get(item.productId)!;
        const unitCost = Number(product.costPrice || product.basePrice || 0);
        const itemTotalAmount = item.quantity * item.unitPrice;
        const itemTotalCost = item.quantity * unitCost;
        const itemProfit = itemTotalAmount - itemTotalCost;

        totalAmount += itemTotalAmount;
        totalCost += itemTotalCost;
        totalProfit += itemProfit;

        return {
          productId: item.productId,
          quantity: item.quantity,
          unitPrice: item.unitPrice,
          unitCost: unitCost,
          totalPrice: itemTotalAmount,
          totalCost: itemTotalCost,
          profit: itemProfit,
        };
      });

      const saleNumber = await this.generateSaleNumber(tx);

      const sale = await tx.sale.create({
        data: {
          saleNumber,
          customerId: input.customerId,
          status: 'draft',
          totalAmount,
          totalCost,
          totalProfit,
          notes: input.notes,
          createdBy: userId,
          items: {
            create: saleItems,
          },
        },
        include: {
          items: {
            include: { product: { select: { id: true, sku: true, name: true } } },
          },
        },
      });

      return sale;
    });
  }

  // ─── Batch Create Sales ───────────────────────────────────────────────────
  async createBatch(input: BatchCreateSalesInput, userId: string) {
    return prisma.$transaction(async (tx) => {
      const createdSales = [];
      const salesDate = new Date();

      for (const saleInput of input.sales) {
        const productIds = saleInput.items.map((i) => i.productId);
        const products = await tx.product.findMany({ where: { id: { in: productIds } } });
        
        if (products.length !== productIds.length) {
          throw new ValidationError('One or more product IDs are invalid in the batch');
        }

        const productMap = new Map(products.map(p => [p.id, p]));

        let totalAmount = 0;
        let totalCost = 0;
        let totalProfit = 0;

        const saleItems = saleInput.items.map((item) => {
          const product = productMap.get(item.productId)!;
          const unitCost = Number(product.costPrice || product.basePrice || 0);
          const itemTotalAmount = item.quantity * item.unitPrice;
          const itemTotalCost = item.quantity * unitCost;
          const itemProfit = itemTotalAmount - itemTotalCost;

          totalAmount += itemTotalAmount;
          totalCost += itemTotalCost;
          totalProfit += itemProfit;

          return {
            productId: item.productId,
            quantity: item.quantity,
            unitPrice: item.unitPrice,
            unitCost: unitCost,
            totalPrice: itemTotalAmount,
            totalCost: itemTotalCost,
            profit: itemProfit,
          };
        });

        // Batch sales are typically completed immediately
        const saleNumber = await this.generateSaleNumber(tx);

        const sale = await tx.sale.create({
          data: {
            saleNumber,
            customerId: saleInput.customerId,
            status: 'completed',
            saleDate: salesDate,
            totalAmount,
            totalCost,
            totalProfit,
            notes: saleInput.notes,
            createdBy: userId,
            items: {
              create: saleItems,
            },
          },
          include: {
            items: true,
          },
        });

        // Deduct inventory
        for (const item of saleItems) {
          const inventory = await tx.inventory.findUnique({
            where: { productId: item.productId },
          });

          if (!inventory) {
            throw new ValidationError(`Inventory record not found for product ${item.productId}`);
          }

          if (inventory.quantity < item.quantity) {
            throw new ValidationError(`Insufficient stock for product ${item.productId}. Available: ${inventory.quantity}, Required: ${item.quantity}`);
          }

          const qtyBefore = inventory.quantity;
          const qtyAfter = qtyBefore - item.quantity;

          await tx.inventory.update({
            where: { productId: item.productId },
            data: { quantity: qtyAfter, lastSoldAt: salesDate },
          });

          await tx.stockMovement.create({
            data: {
              productId: item.productId,
              inventoryId: inventory.id,
              movementType: MovementType.OUT,
              quantity: item.quantity,
              quantityBefore: qtyBefore,
              quantityAfter: qtyAfter,
              reason: `Batch Sale ${saleNumber}`,
              referenceType: 'sale',
              referenceId: sale.id,
              performedBy: userId,
            },
          });
        }

        createdSales.push(sale);
      }

      // Recalculate metrics for the day
      await this.recalculateDailyMetrics(salesDate, tx);

      return createdSales;
    });
  }

  // ─── Update Sale Status (Complete/Cancel) ───────────────────────────────
  async updateStatus(id: string, status: 'completed' | 'cancelled' | 'refunded', userId: string) {
    return prisma.$transaction(async (tx) => {
      const sale = await tx.sale.findUnique({
        where: { id },
        include: { items: true },
      });

      if (!sale) throw new NotFoundError('Sale', id);

      const validTransitions: Record<string, string[]> = {
        draft: ['completed', 'cancelled'],
        completed: ['refunded'],
      };

      const allowed = validTransitions[sale.status] || [];
      if (!allowed.includes(status)) {
        throw new ValidationError(
          `Cannot transition from '${sale.status}' to '${status}'. Allowed: ${allowed.join(', ')}`
        );
      }

      const updatedSale = await tx.sale.update({
        where: { id },
        data: { status },
        include: {
          items: { include: { product: { select: { id: true, sku: true, name: true } } } },
        },
      });

      if (status === 'completed') {
        // Deduct inventory
        for (const item of sale.items) {
          const inventory = await tx.inventory.findUnique({
            where: { productId: item.productId },
          });

          if (!inventory) {
            throw new ValidationError(`Inventory record not found for product ${item.productId}`);
          }

          if (inventory.quantity < item.quantity) {
            throw new ValidationError(`Insufficient stock for product ${item.productId}. Available: ${inventory.quantity}, Required: ${item.quantity}`);
          }

          const qtyBefore = inventory.quantity;
          const qtyAfter = qtyBefore - item.quantity;

          await tx.inventory.update({
            where: { productId: item.productId },
            data: { quantity: qtyAfter, lastSoldAt: new Date() },
          });

          await tx.stockMovement.create({
            data: {
              productId: item.productId,
              inventoryId: inventory.id,
              movementType: MovementType.OUT,
              quantity: item.quantity,
              quantityBefore: qtyBefore,
              quantityAfter: qtyAfter,
              reason: `Sale ${sale.saleNumber} completed`,
              referenceType: 'sale',
              referenceId: sale.id,
              performedBy: userId,
            },
          });
        }
      } else if (status === 'refunded') {
        // Return inventory
        for (const item of sale.items) {
          const inventory = await tx.inventory.findUnique({
            where: { productId: item.productId },
          });

          if (inventory) {
            const qtyBefore = inventory.quantity;
            const qtyAfter = qtyBefore + item.quantity;

            await tx.inventory.update({
              where: { productId: item.productId },
              data: { quantity: qtyAfter },
            });

            await tx.stockMovement.create({
              data: {
                productId: item.productId,
                inventoryId: inventory.id,
                movementType: MovementType.RETURN,
                quantity: item.quantity,
                quantityBefore: qtyBefore,
                quantityAfter: qtyAfter,
                reason: `Sale ${sale.saleNumber} refunded`,
                referenceType: 'sale',
                referenceId: sale.id,
                performedBy: userId,
              },
            });
          }
        }
      }

      // Recalculate metrics for the sale's date
      if (status === 'completed' || status === 'refunded') {
        await this.recalculateDailyMetrics(sale.saleDate, tx);
      }

      return updatedSale;
    });
  }
}

export const salesService = new SalesService();
