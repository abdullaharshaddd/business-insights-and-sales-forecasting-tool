import { Router, Request, Response, NextFunction } from 'express';
import { authService } from './auth.service';
import { registerSchema, loginSchema } from './auth.schema';
import { validate } from '../../middleware/validate.middleware';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { sendSuccess } from '../../shared/utils/response';

const router = Router();

// POST /auth/register — Admin only
router.post(
  '/register',
  authenticate,
  authorize('admin'),
  validate(registerSchema),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = await authService.register(req.body);
      sendSuccess(res, user, 201);
    } catch (err) { next(err); }
  }
);

// POST /auth/login
router.post(
  '/login',
  validate(loginSchema),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const result = await authService.login(req.body);
      sendSuccess(res, result);
    } catch (err) { next(err); }
  }
);

// POST /auth/refresh
router.post(
  '/refresh',
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { refreshToken } = req.body;
      const result = await authService.refreshToken(refreshToken);
      sendSuccess(res, result);
    } catch (err) { next(err); }
  }
);

// GET /auth/me
router.get(
  '/me',
  authenticate,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const profile = await authService.getProfile(req.user!.userId);
      sendSuccess(res, profile);
    } catch (err) { next(err); }
  }
);

export default router;
