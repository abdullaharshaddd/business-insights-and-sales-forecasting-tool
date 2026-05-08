import { Request, Response, NextFunction } from 'express';
import prisma from '../config/database';

export function auditLog(action: string, entityType: string) {
  return async (req: Request, _res: Response, next: NextFunction) => {
    // Store original json method to capture response
    const originalJson = _res.json.bind(_res);
    _res.json = function (body: any) {
      // Only log on successful responses
      if (_res.statusCode >= 200 && _res.statusCode < 300 && req.user) {
        const entityId = req.params.id || body?.data?.id || null;
        prisma.auditLog.create({
          data: {
            userId: req.user.userId,
            action,
            entityType,
            entityId,
            newValues: req.body || null,
            ipAddress: req.ip || req.socket.remoteAddress || null,
          },
        }).catch((err: Error) => console.error('[AuditLog] Error:', err.message));
      }
      return originalJson(body);
    };
    next();
  };
}
