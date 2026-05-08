export const PAGINATION = {
  DEFAULT_PAGE: 1,
  DEFAULT_LIMIT: 20,
  MAX_LIMIT: 100,
};

export const SKU_PREFIX = 'BISFT';

export const STOCK_REASONS = {
  PURCHASE_ORDER: 'purchase_order',
  MANUAL_ADD: 'manual_addition',
  MANUAL_REMOVE: 'manual_removal',
  ADJUSTMENT: 'stock_adjustment',
  ORDER_FULFILLMENT: 'order_fulfillment',
  RETURN: 'customer_return',
  DAMAGED: 'damaged_goods',
  EXPIRED: 'expired_stock',
} as const;
