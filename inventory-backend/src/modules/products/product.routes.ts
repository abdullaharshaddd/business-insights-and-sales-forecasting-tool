import { Router, Request, Response, NextFunction } from 'express';
import { productService } from './product.service';
import { createProductSchema, updateProductSchema, updateStatusSchema } from './product.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { auditLog } from '../../middleware/audit.middleware';
import { sendSuccess, sendPaginated } from '../../shared/utils/response';

const router = Router();

// GET /products — List with search/filter/pagination
router.get('/', authenticate, async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { products, total, page, limit } = await productService.findAll(req.query as any);
    sendPaginated(res, products, total, page, limit);
  } catch (err) { next(err); }
});

// GET /products/low-stock
router.get('/low-stock', authenticate, authorize('admin', 'manager', 'staff'),
  async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const products = await productService.findLowStock();
      sendSuccess(res, products);
    } catch (err) { next(err); }
  }
);

// GET /products/:id
router.get('/:id', authenticate, async (req: Request, res: Response, next: NextFunction) => {
  try {
    const product = await productService.findById(req.params.id);
    sendSuccess(res, product);
  } catch (err) { next(err); }
});

// POST /products
router.post('/', authenticate, authorize('admin', 'manager'), validate(createProductSchema), auditLog('CREATE', 'product'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const product = await productService.create(req.body);
      sendSuccess(res, product, 201);
    } catch (err) { next(err); }
  }
);

// PATCH /products/:id
router.patch('/:id', authenticate, authorize('admin', 'manager'), validate(updateProductSchema), auditLog('UPDATE', 'product'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const product = await productService.update(req.params.id, req.body);
      sendSuccess(res, product);
    } catch (err) { next(err); }
  }
);

// PATCH /products/:id/status
router.patch('/:id/status', authenticate, authorize('admin', 'manager'), validate(updateStatusSchema), auditLog('STATUS_CHANGE', 'product'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const product = await productService.updateStatus(req.params.id, req.body.status);
      sendSuccess(res, product);
    } catch (err) { next(err); }
  }
);

export default router;
