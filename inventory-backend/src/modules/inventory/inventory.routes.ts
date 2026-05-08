import { Router, Request, Response, NextFunction } from 'express';
import { inventoryService } from './inventory.service';
import { addStockSchema, removeStockSchema, adjustStockSchema } from './inventory.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { auditLog } from '../../middleware/audit.middleware';
import { sendSuccess, sendPaginated } from '../../shared/utils/response';

const router = Router();

// GET /inventory — List all stock levels
router.get('/', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { items, total, page, limit } = await inventoryService.findAll(req.query);
      sendPaginated(res, items, total, page, limit);
    } catch (err) { next(err); }
  }
);

// GET /inventory/alerts — Low stock alerts
router.get('/alerts', authenticate, authorize('admin', 'manager', 'staff'),
  async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const alerts = await inventoryService.getLowStockAlerts();
      sendSuccess(res, { alerts, count: alerts.length });
    } catch (err) { next(err); }
  }
);

// GET /inventory/:productId — Stock for a specific product
router.get('/:productId', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const inventory = await inventoryService.findByProductId(req.params.productId);
      sendSuccess(res, inventory);
    } catch (err) { next(err); }
  }
);

// POST /inventory/add-stock — Add stock
router.post('/add-stock', authenticate, authorize('admin', 'manager'),
  validate(addStockSchema), auditLog('STOCK_ADD', 'inventory'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const result = await inventoryService.addStock(req.body, req.user!.userId);
      sendSuccess(res, result, 201);
    } catch (err) { next(err); }
  }
);

// POST /inventory/remove-stock — Remove stock
router.post('/remove-stock', authenticate, authorize('admin', 'manager'),
  validate(removeStockSchema), auditLog('STOCK_REMOVE', 'inventory'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const result = await inventoryService.removeStock(req.body, req.user!.userId);
      sendSuccess(res, result, 201);
    } catch (err) { next(err); }
  }
);

// POST /inventory/adjust — Adjust stock (correction)
router.post('/adjust', authenticate, authorize('admin', 'manager'),
  validate(adjustStockSchema), auditLog('STOCK_ADJUST', 'inventory'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const result = await inventoryService.adjustStock(req.body, req.user!.userId);
      sendSuccess(res, result, 201);
    } catch (err) { next(err); }
  }
);

// GET /inventory/movements/:productId — Movement history
router.get('/movements/:productId', authenticate, authorize('admin', 'manager', 'staff'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { movements, total, page, limit } = await inventoryService.getMovements(
        req.params.productId, req.query
      );
      sendPaginated(res, movements, total, page, limit);
    } catch (err) { next(err); }
  }
);

export default router;
