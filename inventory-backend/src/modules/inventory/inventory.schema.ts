import { z } from 'zod';

export const addStockSchema = z.object({
  productId: z.string().uuid('Invalid product ID'),
  quantity: z.number().int().positive('Quantity must be a positive integer'),
  reason: z.string().min(1, 'Reason is required').max(255),
  referenceType: z.string().max(50).optional(),
  referenceId: z.string().uuid().optional(),
});

export const removeStockSchema = z.object({
  productId: z.string().uuid('Invalid product ID'),
  quantity: z.number().int().positive('Quantity must be a positive integer'),
  reason: z.string().min(1, 'Reason is required').max(255),
  referenceType: z.string().max(50).optional(),
  referenceId: z.string().uuid().optional(),
});

export const adjustStockSchema = z.object({
  productId: z.string().uuid('Invalid product ID'),
  newQuantity: z.number().int().min(0, 'Quantity cannot be negative'),
  reason: z.string().min(1, 'Reason is required').max(255),
});

export type AddStockInput = z.infer<typeof addStockSchema>;
export type RemoveStockInput = z.infer<typeof removeStockSchema>;
export type AdjustStockInput = z.infer<typeof adjustStockSchema>;
