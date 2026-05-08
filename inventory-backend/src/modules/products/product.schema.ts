import { z } from 'zod';

export const createProductSchema = z.object({
  sku: z.string().min(1).max(50),
  barcode: z.string().max(50).optional().nullable(),
  name: z.string().min(1, 'Product name is required').max(255),
  description: z.string().optional(),
  categoryId: z.string().uuid().optional().nullable(),
  supplierId: z.string().uuid().optional().nullable(),
  status: z.enum(['active', 'inactive', 'archived', 'discontinued']).optional().default('active'),
  basePrice: z.number().min(0, 'Price cannot be negative'),
  costPrice: z.number().min(0).optional().nullable(),
  weightG: z.number().min(0).optional().nullable(),
  lengthCm: z.number().min(0).optional().nullable(),
  heightCm: z.number().min(0).optional().nullable(),
  widthCm: z.number().min(0).optional().nullable(),
  photosQty: z.number().int().min(0).optional().default(0),
  reorderPoint: z.number().int().min(0).optional().default(10),
  reorderQty: z.number().int().min(0).optional().default(50),
});

export const updateProductSchema = createProductSchema.partial();

export const updateStatusSchema = z.object({
  status: z.enum(['active', 'inactive', 'archived', 'discontinued']),
});

export const productQuerySchema = z.object({
  page: z.string().optional(),
  limit: z.string().optional(),
  search: z.string().optional(),
  categoryId: z.string().uuid().optional(),
  supplierId: z.string().uuid().optional(),
  status: z.enum(['active', 'inactive', 'archived', 'discontinued']).optional(),
  sortBy: z.enum(['name', 'basePrice', 'createdAt', 'sku']).optional().default('createdAt'),
  order: z.enum(['asc', 'desc']).optional().default('desc'),
});

export type CreateProductInput = z.infer<typeof createProductSchema>;
export type UpdateProductInput = z.infer<typeof updateProductSchema>;
