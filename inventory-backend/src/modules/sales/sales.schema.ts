import { z } from 'zod';

export const saleItemSchema = z.object({
  productId: z.string().uuid('Invalid product ID'),
  quantity: z.number().int().positive('Quantity must be greater than 0'),
  unitPrice: z.number().positive('Unit price must be greater than 0'),
});

export const createSaleSchema = z.object({
  body: z.object({
    customerId: z.string().optional(),
    items: z.array(saleItemSchema).min(1, 'Sale must have at least one item'),
    notes: z.string().optional(),
  }),
});

export const batchCreateSalesSchema = z.object({
  body: z.object({
    sales: z.array(
      z.object({
        customerId: z.string().optional(),
        items: z.array(saleItemSchema).min(1),
        notes: z.string().optional(),
      })
    ).min(1, 'Batch must contain at least one sale'),
  }),
});

export const updateSaleStatusSchema = z.object({
  body: z.object({
    status: z.enum(['completed', 'cancelled', 'refunded']),
  }),
});

export type CreateSaleInput = z.infer<typeof createSaleSchema>['body'];
export type BatchCreateSalesInput = z.infer<typeof batchCreateSalesSchema>['body'];
export type SaleItemInput = z.infer<typeof saleItemSchema>;
