import prisma from '../../config/database';
import { CreateSaleInput, BatchCreateSalesInput, SaleItemInput } from './sales.schema';
import { NotFoundError, ValidationError, InsufficientStockError } from '../../shared/errors/AppError';
import { SaleStatus, MovementType, Prisma } from '@prisma/client';
import { parsePagination } from '../../shared/utils/response';

type ProductPricing = { costPrice: Prisma.Decimal | null; basePrice: Prisma.Decimal };

type SaleItemRow = {
  productId: string;
  quantity: number;
  unitPrice: number;
  unitCost: number;
  totalPrice: number;
  totalCost: number;
  profit: number;
};

// ─── Generate Sale number ─────────────────────────────────────────────────────
async function generateSaleNumber(tx: Prisma.TransactionClient): Promise<string> {
  const year = new Date().getFullYear();
  const count = await tx.sale.count({
    where: {
      createdAt: { gte: new Date(`${year}-01-01`), lt: new Date(`${year + 1}-01-01`) },
    },
  });
  return `SAL-${year}-${String(count + 1).padStart(5, '0')}`;
}

// ─── Build sale item rows from inputs + product map ───────────────────────────
function buildSaleItems(
  items: SaleItemInput[],
  productMap: Map<string, ProductPricing>
): { rows: SaleItemRow[]; totalAmount: number; totalCost: number; totalProfit: number } {
  let totalAmount = 0;
  let totalCost = 0;

  const rows = items.map((item): SaleItemRow => {
    const product = productMap.get(item.productId)!;
    const unitCost = Number(product.costPrice ?? product.basePrice);
    const itemTotalPrice = item.quantity * item.unitPrice;
    const itemTotalCost = item.quantity * unitCost;

    totalAmount += itemTotalPrice;
    totalCost += itemTotalCost;

    return {
      productId: item.productId,
      quantity: item.quantity,
      unitPrice: item.unitPrice,
      unitCost,
      totalPrice: itemTotalPrice,
      totalCost: itemTotalCost,
      profit: itemTotalPrice - itemTotalCost,
    };
  });

  return { rows, totalAmount, totalCost, totalProfit: totalAmount - totalCost };
}

// ─── Deduct inventory + record stock movements ────────────────────────────────
async function deductInventory(
  saleItems: Array<{ productId: string; quantity: number }>,
  saleId: string,
  saleNumber: string,
  userId: string,
  tx: Prisma.TransactionClient,
  reason?: string
): Promise<void> {
  for (const item of saleItems) {
    const inventory = await tx.inventory.findUnique({ where: { productId: item.productId } });
    if (!inventory) {
      throw new ValidationError(`Inventory record not found for product ${item.productId}`);
    }

    const available = inventory.quantity - inventory.reservedQty;
    if (available < item.quantity) {
      throw new InsufficientStockError(available, item.quantity);
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
        reason: reason ?? `Sale ${saleNumber} completed`,
        referenceType: 'sale',
        referenceId: saleId,
        performedBy: userId,
      },
    });
  }
}

// ─── Return inventory + record stock movements ────────────────────────────────
async function returnInventory(
  saleItems: Array<{ productId: string; quantity: number }>,
  saleId: string,
  saleNumber: string,
  userId: string,
  tx: Prisma.TransactionClient
): Promise<void> {
  for (const item of saleItems) {
    const inventory = await tx.inventory.findUnique({ where: { productId: item.productId } });
    if (!inventory) continue;

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
        reason: `Sale ${saleNumber} refunded`,
        referenceType: 'sale',
        referenceId: saleId,
        performedBy: userId,
      },
    });
  }
}

export class SalesService {
  // ─── Metrics Recalculation Engine (single day) ─────────────────────────────
  async recalculateDailyMetrics(date: Date, tx?: Prisma.TransactionClient): Promise<void> {
    const db = tx ?? prisma;
    const startOfDay = new Date(date);
    startOfDay.setUTCHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setUTCHours(23, 59, 59, 999);

    const sales = await db.sale.findMany({
      where: { status: 'completed', saleDate: { gte: startOfDay, lte: endOfDay } },
      select: { totalAmount: true, totalCost: true, totalProfit: true },
    });

    const totalSales = sales.length;
    const revenue = sales.reduce((sum: number, s) => sum + Number(s.totalAmount), 0);
    const cost = sales.reduce((sum: number, s) => sum + Number(s.totalCost), 0);
    const profit = sales.reduce((sum: number, s) => sum + Number(s.totalProfit), 0);
    const marginPct = revenue > 0 ? (profit / revenue) * 100 : 0;

    await db.salesMetrics.upsert({
      where: { date: startOfDay },
      update: { totalSales, revenue, cost, profit, marginPct },
      create: { date: startOfDay, totalSales, revenue, cost, profit, marginPct },
    });
  }

  // ─── Metrics Recalculation — date range ────────────────────────────────────
  async recalculateDateRange(startDate: string, endDate?: string) {
    const start = new Date(startDate);
    start.setUTCHours(0, 0, 0, 0);
    const end = endDate ? new Date(endDate) : new Date(startDate);
    end.setUTCHours(0, 0, 0, 0);

    if (end < start) {
      throw new ValidationError('endDate must be on or after startDate');
    }

    const dates: Date[] = [];
    const cursor = new Date(start);
    while (cursor <= end) {
      dates.push(new Date(cursor));
      cursor.setUTCDate(cursor.getUTCDate() + 1);
    }

    for (const date of dates) {
      await this.recalculateDailyMetrics(date);
    }

    return { datesProcessed: dates.length, startDate, endDate: endDate ?? startDate };
  }

  // ─── Get Metrics ───────────────────────────────────────────────────────────
  async getMetrics(query: Record<string, any>) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const where: Prisma.SalesMetricsWhereInput = {};
    if (query.startDate || query.endDate) {
      const dateFilter: Prisma.DateTimeFilter = {};
      if (query.startDate) dateFilter.gte = new Date(String(query.startDate));
      if (query.endDate) dateFilter.lte = new Date(String(query.endDate));
      where.date = dateFilter;
    }

    const [metrics, total] = await Promise.all([
      prisma.salesMetrics.findMany({ where, orderBy: { date: 'desc' }, skip, take: limit }),
      prisma.salesMetrics.count({ where }),
    ]);

    return { metrics, total, page, limit };
  }

  // ─── List Sales ────────────────────────────────────────────────────────────
  async findAll(query: Record<string, any>) {
    const { page, limit } = parsePagination(query);
    const skip = (page - 1) * limit;

    const where: Prisma.SaleWhereInput = {};
    if (query.status) where.status = String(query.status) as SaleStatus;
    if (query.customerId) where.customerId = String(query.customerId);
    if (query.startDate || query.endDate) {
      const dateFilter: Prisma.DateTimeFilter = {};
      if (query.startDate) dateFilter.gte = new Date(String(query.startDate));
      if (query.endDate) dateFilter.lte = new Date(String(query.endDate));
      where.saleDate = dateFilter;
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

  // ─── Get Sale detail ───────────────────────────────────────────────────────
  async findById(id: string) {
    const sale = await prisma.sale.findUnique({
      where: { id },
      include: {
        user: { select: { id: true, fullName: true, email: true } },
        items: { include: { product: { select: { id: true, sku: true, name: true } } } },
      },
    });
    if (!sale) throw new NotFoundError('Sale', id);
    return sale;
  }

  // ─── Create Single Sale ────────────────────────────────────────────────────
  async create(input: CreateSaleInput, userId: string) {
    return prisma.$transaction(async (tx: Prisma.TransactionClient) => {
      const productIds = input.items.map((i) => i.productId);
      const products = await tx.product.findMany({
        where: { id: { in: productIds } },
        select: { id: true, costPrice: true, basePrice: true },
      });

      if (products.length !== productIds.length) {
        throw new ValidationError('One or more product IDs are invalid');
      }

      const productMap = new Map<string, ProductPricing>(
        products.map((p) => [p.id, { costPrice: p.costPrice, basePrice: p.basePrice }])
      );

      const { rows: saleItems, totalAmount, totalCost, totalProfit } = buildSaleItems(input.items, productMap);

      const targetStatus = input.status ?? 'draft';

      // For completed sales, validate stock before any writes
      if (targetStatus === 'completed') {
        for (const item of saleItems) {
          const inv = await tx.inventory.findUnique({ where: { productId: item.productId } });
          if (!inv) throw new ValidationError(`Inventory not found for product ${item.productId}`);
          const available = inv.quantity - inv.reservedQty;
          if (available < item.quantity) throw new InsufficientStockError(available, item.quantity);
        }
      }

      const saleNumber = await generateSaleNumber(tx);

      const sale = await tx.sale.create({
        data: {
          saleNumber,
          customerId: input.customerId,
          status: targetStatus,
          totalAmount,
          totalCost,
          totalProfit,
          notes: input.notes,
          createdBy: userId,
          items: { create: saleItems },
        },
        include: {
          items: { include: { product: { select: { id: true, sku: true, name: true } } } },
        },
      });

      if (targetStatus === 'completed') {
        await deductInventory(saleItems, sale.id, saleNumber, userId, tx);
        await this.recalculateDailyMetrics(sale.saleDate, tx);
      }

      return sale;
    });
  }

  // ─── Batch Create Sales ────────────────────────────────────────────────────
  // Pre-validates ALL inventory concurrently before writing any row.
  async createBatch(input: BatchCreateSalesInput, userId: string) {
    return prisma.$transaction(async (tx: Prisma.TransactionClient) => {
      const salesDate = new Date();

      // Gather all unique product IDs across the entire batch
      const allProductIds = [...new Set(input.sales.flatMap((s) => s.items.map((i) => i.productId)))];

      const products = await tx.product.findMany({
        where: { id: { in: allProductIds } },
        select: { id: true, costPrice: true, basePrice: true },
      });

      if (products.length !== allProductIds.length) {
        throw new ValidationError('One or more product IDs are invalid in the batch');
      }

      const productMap = new Map<string, ProductPricing>(
        products.map((p) => [p.id, { costPrice: p.costPrice, basePrice: p.basePrice }])
      );

      // Build all sale item rows upfront
      const preparedSales = input.sales.map((saleInput) => ({
        saleInput,
        ...buildSaleItems(saleInput.items, productMap),
      }));

      // Aggregate required quantity per product across all sales in the batch
      const required = new Map<string, number>();
      for (const { rows } of preparedSales) {
        for (const row of rows) {
          required.set(row.productId, (required.get(row.productId) ?? 0) + row.quantity);
        }
      }

      // Concurrent inventory pre-validation — fail fast before any writes
      await Promise.all(
        [...required.entries()].map(async ([productId, qty]) => {
          const inv = await tx.inventory.findUnique({ where: { productId } });
          if (!inv) throw new ValidationError(`Inventory not found for product ${productId}`);
          const available = inv.quantity - inv.reservedQty;
          if (available < qty) throw new InsufficientStockError(available, qty);
        })
      );

      // All stock confirmed — write sales, items, and movements
      const createdSales = [];

      for (const { saleInput, rows, totalAmount, totalCost, totalProfit } of preparedSales) {
        const saleNumber = await generateSaleNumber(tx);

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
            items: { create: rows },
          },
          include: { items: true },
        });

        await deductInventory(rows, sale.id, saleNumber, userId, tx, `Batch sale ${saleNumber}`);
        createdSales.push(sale);
      }

      await this.recalculateDailyMetrics(salesDate, tx);

      return createdSales;
    });
  }

  // ─── Update Sale Status ────────────────────────────────────────────────────
  async updateStatus(id: string, status: 'completed' | 'cancelled' | 'refunded', userId: string) {
    return prisma.$transaction(async (tx: Prisma.TransactionClient) => {
      const sale = await tx.sale.findUnique({ where: { id }, include: { items: true } });
      if (!sale) throw new NotFoundError('Sale', id);

      const validTransitions: Record<string, string[]> = {
        draft: ['completed', 'cancelled'],
        completed: ['refunded'],
      };

      const allowed = validTransitions[sale.status] ?? [];
      if (!allowed.includes(status)) {
        throw new ValidationError(
          `Cannot transition from '${sale.status}' to '${status}'. Allowed: ${allowed.join(', ')}`
        );
      }

      // Pre-validate stock before writing when completing
      if (status === 'completed') {
        for (const item of sale.items) {
          const inv = await tx.inventory.findUnique({ where: { productId: item.productId } });
          if (!inv) throw new ValidationError(`Inventory not found for product ${item.productId}`);
          const available = inv.quantity - inv.reservedQty;
          if (available < item.quantity) throw new InsufficientStockError(available, item.quantity);
        }
      }

      const updatedSale = await tx.sale.update({
        where: { id },
        data: { status },
        include: {
          items: { include: { product: { select: { id: true, sku: true, name: true } } } },
        },
      });

      if (status === 'completed') {
        await deductInventory(sale.items, sale.id, sale.saleNumber, userId, tx);
      } else if (status === 'refunded') {
        await returnInventory(sale.items, sale.id, sale.saleNumber, userId, tx);
      }

      if (status === 'completed' || status === 'refunded') {
        await this.recalculateDailyMetrics(sale.saleDate, tx);
      }

      return updatedSale;
    });
  }
}

export const salesService = new SalesService();
