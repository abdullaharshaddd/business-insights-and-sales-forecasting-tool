import { Router, Request, Response, NextFunction } from 'express';
import { purchaseOrderService } from './purchase-order.service';
import { createPurchaseOrderSchema, updatePOStatusSchema, receivePOSchema } from './purchase-order.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { auditLog } from '../../middleware/audit.middleware';
import { sendSuccess, sendPaginated } from '../../shared/utils/response';

const router = Router();

router.get('/', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { orders, total, page, limit } = await purchaseOrderService.findAll(req.query);
      sendPaginated(res, orders, total, page, limit);
    } catch (err) { next(err); }
  }
);

router.get('/:id', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const po = await purchaseOrderService.findById(req.params.id);
      sendSuccess(res, po);
    } catch (err) { next(err); }
  }
);

router.post('/', authenticate, authorize('admin', 'manager'),
  validate(createPurchaseOrderSchema), auditLog('CREATE', 'purchase_order'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const po = await purchaseOrderService.create(req.body, req.user!.userId);
      sendSuccess(res, po, 201);
    } catch (err) { next(err); }
  }
);

router.patch('/:id/status', authenticate, authorize('admin', 'manager'),
  validate(updatePOStatusSchema), auditLog('STATUS_CHANGE', 'purchase_order'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const po = await purchaseOrderService.updateStatus(req.params.id, req.body.status);
      sendSuccess(res, po);
    } catch (err) { next(err); }
  }
);

router.post('/:id/receive', authenticate, authorize('admin', 'manager'),
  validate(receivePOSchema), auditLog('RECEIVE', 'purchase_order'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const result = await purchaseOrderService.receive(req.params.id, req.body, req.user!.userId);
      sendSuccess(res, result);
    } catch (err) { next(err); }
  }
);

export default router;
