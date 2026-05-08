import { Router, Request, Response, NextFunction } from 'express';
import { supplierService } from './supplier.service';
import { createSupplierSchema, updateSupplierSchema } from './supplier.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { auditLog } from '../../middleware/audit.middleware';
import { sendSuccess, sendPaginated } from '../../shared/utils/response';

const router = Router();

router.get('/', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { suppliers, total, page, limit } = await supplierService.findAll(req.query);
      sendPaginated(res, suppliers, total, page, limit);
    } catch (err) { next(err); }
  }
);

router.get('/:id', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const supplier = await supplierService.findById(req.params.id);
      sendSuccess(res, supplier);
    } catch (err) { next(err); }
  }
);

router.post('/', authenticate, authorize('admin', 'manager'),
  validate(createSupplierSchema), auditLog('CREATE', 'supplier'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const supplier = await supplierService.create(req.body);
      sendSuccess(res, supplier, 201);
    } catch (err) { next(err); }
  }
);

router.patch('/:id', authenticate, authorize('admin', 'manager'),
  validate(updateSupplierSchema), auditLog('UPDATE', 'supplier'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const supplier = await supplierService.update(req.params.id, req.body);
      sendSuccess(res, supplier);
    } catch (err) { next(err); }
  }
);

router.delete('/:id', authenticate, authorize('admin'),
  auditLog('DEACTIVATE', 'supplier'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      await supplierService.deactivate(req.params.id);
      sendSuccess(res, { message: 'Supplier deactivated' });
    } catch (err) { next(err); }
  }
);

export default router;
