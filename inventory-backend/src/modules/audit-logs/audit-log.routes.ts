import { Router, Request, Response, NextFunction } from 'express';
import prisma from '../../config/database';
import { authenticate, authorize } from '../../middleware/auth.middleware';
import { parsePagination, sendPaginated } from '../../shared/utils/response';
import { Prisma } from '@prisma/client';

const router = Router();

// GET /audit-logs — List all audit logs (Admin only)
router.get('/', authenticate, authorize('admin'), async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { page, limit } = parsePagination(req.query);
    const skip = (page - 1) * limit;

    const where: Prisma.AuditLogWhereInput = {};

    if (req.query.userId) where.userId = req.query.userId as string;
    if (req.query.entityType) where.entityType = req.query.entityType as string;
    if (req.query.entityId) where.entityId = req.query.entityId as string;
    if (req.query.action) where.action = req.query.action as string;

    const [logs, total] = await Promise.all([
      prisma.auditLog.findMany({
        where,
        include: {
          user: { select: { id: true, fullName: true, email: true } },
        },
        orderBy: { createdAt: 'desc' },
        skip,
        take: limit,
      }),
      prisma.auditLog.count({ where }),
    ]);

    sendPaginated(res, logs, total, page, limit);
  } catch (err) { next(err); }
});

export default router;
