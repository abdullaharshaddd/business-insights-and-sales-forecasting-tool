import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { env } from './config/env';
import { errorHandler } from './middleware/error.middleware';

// Route imports
import authRoutes from './modules/auth/auth.routes';
import categoryRoutes from './modules/categories/category.routes';
import productRoutes from './modules/products/product.routes';
import inventoryRoutes from './modules/inventory/inventory.routes';
import supplierRoutes from './modules/suppliers/supplier.routes';
import purchaseOrderRoutes from './modules/purchase-orders/purchase-order.routes';
import auditLogRoutes from './modules/audit-logs/audit-log.routes';

const app = express();

// ─── Global Middleware ────────────────────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: env.CORS_ORIGIN, credentials: true }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 200,
  message: { success: false, error: { code: 'RATE_LIMIT', message: 'Too many requests' } },
});
app.use('/api/', limiter);

// ─── Health Check ─────────────────────────────────────────────────────────
app.get('/api/v1/health', (_req, res) => {
  res.json({ success: true, data: { status: 'online', timestamp: new Date().toISOString() } });
});

// ─── API Routes ───────────────────────────────────────────────────────────
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/categories', categoryRoutes);
app.use('/api/v1/products', productRoutes);
app.use('/api/v1/inventory', inventoryRoutes);
app.use('/api/v1/suppliers', supplierRoutes);
app.use('/api/v1/purchase-orders', purchaseOrderRoutes);
app.use('/api/v1/audit-logs', auditLogRoutes);

// ─── 404 Handler ──────────────────────────────────────────────────────────
app.use((_req, res) => {
  res.status(404).json({
    success: false,
    error: { code: 'NOT_FOUND', message: 'Endpoint not found' },
  });
});

// ─── Global Error Handler ─────────────────────────────────────────────────
app.use(errorHandler);

export default app;
