import { z } from 'zod';

const poItemSchema = z.object({
  productId: z.string().uuid(),
  quantityOrdered: z.number().int().positive('Quantity must be positive'),
  unitCost: z.number().positive('Unit cost must be positive'),
});

export const createPurchaseOrderSchema = z.object({
  supplierId: z.string().uuid('Invalid supplier ID'),
  expectedDate: z.string().datetime().optional(),
  notes: z.string().optional(),
  items: z.array(poItemSchema).min(1, 'At least one item is required'),
});

export const updatePOStatusSchema = z.object({
  status: z.enum(['submitted', 'confirmed', 'cancelled']),
});

export const receivePOSchema = z.object({
  items: z.array(z.object({
    purchaseOrderItemId: z.string().uuid(),
    quantityReceived: z.number().int().min(0, 'Quantity cannot be negative'),
  })).min(1, 'At least one item must be received'),
});

export type CreatePOInput = z.infer<typeof createPurchaseOrderSchema>;
export type ReceivePOInput = z.infer<typeof receivePOSchema>;
