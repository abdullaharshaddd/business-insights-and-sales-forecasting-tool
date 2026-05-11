import { Router, Request, Response, NextFunction } from 'express';
import { salesService } from './sales.service';
import { createSaleSchema, batchCreateSalesSchema, updateSaleStatusSchema } from './sales.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { auditLog } from '../../middleware/audit.middleware';
import { sendSuccess, sendPaginated } from '../../shared/utils/response';

const router = Router();

router.get('/', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { sales, total, page, limit } = await salesService.findAll(req.query);
      sendPaginated(res, sales, total, page, limit);
    } catch (err) { next(err); }
  }
);

router.get('/:id', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sale = await salesService.findById(req.params.id);
      sendSuccess(res, sale);
    } catch (err) { next(err); }
  }
);

router.post('/', authenticate, authorize('admin', 'manager', 'staff'),
  validate(createSaleSchema), auditLog('CREATE', 'sale'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sale = await salesService.create(req.body, req.user!.userId);
      sendSuccess(res, sale, 201);
    } catch (err) { next(err); }
  }
);

router.post('/batch', authenticate, authorize('admin', 'manager'),
  validate(batchCreateSalesSchema), auditLog('BATCH_CREATE', 'sale'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sales = await salesService.createBatch(req.body, req.user!.userId);
      sendSuccess(res, sales, 201);
    } catch (err) { next(err); }
  }
);

router.patch('/:id/status', authenticate, authorize('admin', 'manager'),
  validate(updateSaleStatusSchema), auditLog('STATUS_CHANGE', 'sale'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sale = await salesService.updateStatus(req.params.id, req.body.status, req.user!.userId);
      sendSuccess(res, sale);
    } catch (err) { next(err); }
  }
);

router.post('/metrics/recalculate', authenticate, authorize('admin', 'manager'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const date = req.body.date ? new Date(req.body.date) : new Date();
      await salesService.recalculateDailyMetrics(date);
      sendSuccess(res, { message: 'Metrics recalculated successfully for ' + date.toISOString() });
    } catch (err) { next(err); }
  }
);

export default router;
