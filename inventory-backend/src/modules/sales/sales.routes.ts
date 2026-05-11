import { Router, Request, Response, NextFunction } from 'express';
import { salesService } from './sales.service';
import {
  createSaleSchema,
  batchCreateSalesSchema,
  updateSaleStatusSchema,
  recalculateMetricsSchema,
} from './sales.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { auditLog } from '../../middleware/audit.middleware';
import { sendSuccess, sendPaginated } from '../../shared/utils/response';

const router = Router();

// GET /sales — Paginated list
router.get('/', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { sales, total, page, limit } = await salesService.findAll(req.query as Record<string, any>);
      sendPaginated(res, sales, total, page, limit);
    } catch (err) { next(err); }
  }
);

// GET /sales/metrics — Query aggregated daily metrics (must be before /:id)
router.get('/metrics', authenticate, authorize('admin', 'manager'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { metrics, total, page, limit } = await salesService.getMetrics(req.query as Record<string, any>);
      sendPaginated(res, metrics, total, page, limit);
    } catch (err) { next(err); }
  }
);

// GET /sales/:id — Sale detail
router.get('/:id', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sale = await salesService.findById(req.params.id as string);
      sendSuccess(res, sale);
    } catch (err) { next(err); }
  }
);

// POST /sales — Record single sale (draft or completed)
router.post('/', authenticate, authorize('admin', 'manager', 'staff'),
  validate(createSaleSchema), auditLog('CREATE', 'sale'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sale = await salesService.create(req.body, req.user!.userId);
      sendSuccess(res, sale, 201);
    } catch (err) { next(err); }
  }
);

// POST /sales/batch — Batch recording (all completed with concurrent stock validation)
router.post('/batch', authenticate, authorize('admin', 'manager'),
  validate(batchCreateSalesSchema), auditLog('BATCH_CREATE', 'sale'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sales = await salesService.createBatch(req.body, req.user!.userId);
      sendSuccess(res, sales, 201);
    } catch (err) { next(err); }
  }
);

// PATCH /sales/:id/status — Status transition
router.patch('/:id/status', authenticate, authorize('admin', 'manager'),
  validate(updateSaleStatusSchema), auditLog('STATUS_CHANGE', 'sale'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const sale = await salesService.updateStatus(
        req.params.id as string,
        req.body.status,
        req.user!.userId
      );
      sendSuccess(res, sale);
    } catch (err) { next(err); }
  }
);

// POST /sales/metrics/recalculate — On-demand recalculation for a date range
router.post('/metrics/recalculate', authenticate, authorize('admin', 'manager'),
  validate(recalculateMetricsSchema),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const result = await salesService.recalculateDateRange(req.body.startDate, req.body.endDate);
      sendSuccess(res, result);
    } catch (err) { next(err); }
  }
);

export default router;
