import { z } from 'zod';

export const createSupplierSchema = z.object({
  name: z.string().min(1, 'Name is required').max(255),
  email: z.string().email().optional().nullable(),
  phone: z.string().max(50).optional().nullable(),
  contactPerson: z.string().max(100).optional().nullable(),
  city: z.string().max(100).optional().nullable(),
  state: z.string().max(10).optional().nullable(),
  zipCode: z.string().max(20).optional().nullable(),
  paymentTerms: z.string().max(100).optional().nullable(),
  leadTimeDays: z.number().int().min(0).optional().default(7),
  notes: z.string().optional().nullable(),
});

export const updateSupplierSchema = createSupplierSchema.partial();

export type CreateSupplierInput = z.infer<typeof createSupplierSchema>;
export type UpdateSupplierInput = z.infer<typeof updateSupplierSchema>;
