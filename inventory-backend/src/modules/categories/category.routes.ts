import { Router, Request, Response, NextFunction } from 'express';
import { categoryService } from './category.service';
import { createCategorySchema, updateCategorySchema } from './category.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { auditLog } from '../../middleware/audit.middleware';
import { sendSuccess } from '../../shared/utils/response';

const router = Router();

router.get('/', authenticate, async (_req: Request, res: Response, next: NextFunction) => {
  try {
    const categories = await categoryService.findAll();
    sendSuccess(res, categories);
  } catch (err) { next(err); }
});

router.get('/:id', authenticate, async (req: Request, res: Response, next: NextFunction) => {
  try {
    const category = await categoryService.findById(req.params.id);
    sendSuccess(res, category);
  } catch (err) { next(err); }
});

router.post('/', authenticate, authorize('admin'), validate(createCategorySchema), auditLog('CREATE', 'category'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const category = await categoryService.create(req.body);
      sendSuccess(res, category, 201);
    } catch (err) { next(err); }
  }
);

router.patch('/:id', authenticate, authorize('admin'), validate(updateCategorySchema), auditLog('UPDATE', 'category'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const category = await categoryService.update(req.params.id, req.body);
      sendSuccess(res, category);
    } catch (err) { next(err); }
  }
);

router.delete('/:id', authenticate, authorize('admin'), auditLog('DELETE', 'category'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      await categoryService.softDelete(req.params.id);
      sendSuccess(res, { message: 'Category deactivated' });
    } catch (err) { next(err); }
  }
);

export default router;
