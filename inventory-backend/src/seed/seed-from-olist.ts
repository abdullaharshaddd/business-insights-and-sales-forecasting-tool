/**
 * seed-from-olist.ts
 * ==================
 * Full migration script that reads the Olist SQLite database and seeds
 * the PostgreSQL database with ALL tables (Historical + Operational).
 */

import path from 'path';
import bcrypt from 'bcryptjs';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const OLIST_DB_PATH = path.resolve(__dirname, '../../../data/processed/olist/olist.db');
const SKU_PREFIX = 'BISFT';

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

function generateSKU(index: number): string {
  return `${SKU_PREFIX}-${String(index).padStart(6, '0')}`;
}

// Helper to batch inserts for performance
async function batchInsert(tx: any, model: string, data: any[], batchSize = 500) {
  for (let i = 0; i < data.length; i += batchSize) {
    const batch = data.slice(i, i + batchSize);
    await (tx as any)[model].createMany({ data: batch, skipDuplicates: true });
    if (i % (batchSize * 10) === 0 && i > 0) console.log(`    ... inserted ${i} rows into ${model}`);
  }
}

async function seed() {
  console.log('='.repeat(55));
  console.log('  BISFT Unified Database — Full Migration');
  console.log('='.repeat(55));

  const Database = require('better-sqlite3');
  const sqlite = new Database(OLIST_DB_PATH, { readonly: true });

  try {
    // 1. Admin User
    console.log('\n[1/10] Creating admin user...');
    const hash = await bcrypt.hash(process.env.ADMIN_PASSWORD || 'Admin@123', 12);
    await prisma.user.upsert({
      where: { email: process.env.ADMIN_EMAIL || 'admin@bisft.com' },
      update: {},
      create: {
        email: process.env.ADMIN_EMAIL || 'admin@bisft.com',
        passwordHash: hash,
        fullName: 'System Admin',
        role: 'admin',
      },
    });

    // 2. Categories
    console.log('\n[2/10] Migrating categories...');
    const catRows = sqlite.prepare('SELECT * FROM product_category_name_translation').all();
    for (const row of catRows) {
      await prisma.category.upsert({
        where: { slug: slugify(row.product_category_name_english) },
        update: {},
        create: {
          name: row.product_category_name_english.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()),
          namePt: row.product_category_name,
          slug: slugify(row.product_category_name_english),
        },
      });
    }

    // 3. Customers
    console.log('\n[3/10] Migrating customers...');
    const customers = sqlite.prepare('SELECT * FROM customers').all().map((r: any) => ({
      id: r.customer_id,
      uniqueId: r.customer_unique_id,
      zipCodePrefix: r.customer_zip_code_prefix,
      city: r.customer_city,
      state: r.customer_state,
    }));
    await batchInsert(prisma, 'customer', customers);

    // 4. Suppliers (Sellers)
    console.log('\n[4/10] Migrating suppliers (sellers)...');
    const sellers = sqlite.prepare('SELECT * FROM sellers').all().map((r: any) => ({
      id: r.seller_id, // We use the Olist seller_id as the primary key
      legacySellerId: r.seller_id,
      name: `Seller ${r.seller_city} (${r.seller_state})`,
      city: r.seller_city,
      state: r.seller_state,
      zipCode: String(r.seller_zip_code_prefix),
    }));
    await batchInsert(prisma, 'supplier', sellers);

    // 5. Products
    console.log('\n[5/10] Migrating products...');
    const productRows = sqlite.prepare('SELECT * FROM products').all();
    const catMap = new Map((await prisma.category.findMany()).map(c => [c.namePt, c.id]));
    
    // Get average prices from order_items
    const avgPrices = new Map(sqlite.prepare('SELECT product_id, AVG(price) as p FROM order_items GROUP BY 1').all().map((r: any) => [r.product_id, r.p]));

    const products = productRows.map((r: any, i: number) => ({
      id: r.product_id,
      legacyId: r.product_id,
      sku: generateSKU(i + 1),
      name: `${r.product_category_name || 'General'} Product`,
      categoryId: catMap.get(r.product_category_name) || null,
      basePrice: avgPrices.get(r.product_id) || 0,
      weightG: r.product_weight_g,
      lengthCm: r.product_length_cm,
      heightCm: r.product_height_cm,
      widthCm: r.product_width_cm,
      photosQty: Math.round(r.product_photos_qty || 0),
    }));
    await batchInsert(prisma, 'product', products);

    // 6. Inventory Initial Stock
    console.log('\n[6/10] Initializing inventory...');
    const inventory = products.map((p: any) => ({
      productId: p.id,
      quantity: Math.floor(Math.random() * 100) + 10, // Randomized for demo
    }));
    await batchInsert(prisma, 'inventory', inventory);

    // 7. Orders
    console.log('\n[7/10] Migrating orders...');
    const orders = sqlite.prepare('SELECT * FROM orders').all().map((r: any) => ({
      id: r.order_id,
      customerId: r.customer_id,
      status: r.order_status,
      purchaseTimestamp: r.order_purchase_timestamp ? new Date(r.order_purchase_timestamp) : null,
      approvedAt: r.order_approved_at ? new Date(r.order_approved_at) : null,
      deliveredCarrierDate: r.order_delivered_carrier_date ? new Date(r.order_delivered_carrier_date) : null,
      deliveredCustomerDate: r.order_delivered_customer_date ? new Date(r.order_delivered_customer_date) : null,
      estimatedDeliveryDate: r.order_estimated_delivery_date ? new Date(r.order_estimated_delivery_date) : null,
    }));
    await batchInsert(prisma, 'order', orders);

    // 8. Order Items
    console.log('\n[8/10] Migrating order items...');
    const orderItems = sqlite.prepare('SELECT * FROM order_items').all().map((r: any) => ({
      orderId: r.order_id,
      orderItemId: r.order_item_id,
      productId: r.product_id,
      sellerId: r.seller_id,
      shippingLimitDate: r.shipping_limit_date ? new Date(r.shipping_limit_date) : null,
      price: r.price,
      freightValue: r.freight_value,
    }));
    await batchInsert(prisma, 'orderItem', orderItems);

    // 9. Order Payments & Reviews
    console.log('\n[9/10] Migrating payments and reviews...');
    const payments = sqlite.prepare('SELECT * FROM order_payments').all().map((r: any) => ({
      orderId: r.order_id,
      paymentSequential: r.payment_sequential,
      paymentType: r.payment_type,
      paymentInstallments: r.payment_installments,
      paymentValue: r.payment_value,
    }));
    await batchInsert(prisma, 'orderPayment', payments);

    const reviews = sqlite.prepare('SELECT * FROM order_reviews').all().map((r: any) => ({
      id: r.review_id,
      orderId: r.order_id,
      reviewScore: r.review_score,
      reviewCommentTitle: r.review_comment_title,
      reviewCommentMessage: r.review_comment_message,
      reviewCreationDate: r.review_creation_date ? new Date(r.review_creation_date) : null,
      reviewAnswerTimestamp: r.review_answer_timestamp ? new Date(r.review_answer_timestamp) : null,
    }));
    await batchInsert(prisma, 'orderReview', reviews);

    // 10. Geolocation
    console.log('\n[10/10] Migrating geolocation (this may take a while)...');
    const geo = sqlite.prepare('SELECT * FROM geolocation').all().map((r: any) => ({
      zipCodePrefix: r.geolocation_zip_code_prefix,
      lat: r.geolocation_lat,
      lng: r.geolocation_lng,
      city: r.geolocation_city,
      state: r.geolocation_state,
    }));
    await batchInsert(prisma, 'geolocation', geo, 1000);

    console.log('\n✅ UNIFIED DATABASE MIGRATION COMPLETE!');
  } catch (err) {
    console.error('❌ Migration failed:', err);
  } finally {
    sqlite.close();
    await prisma.$disconnect();
  }
}

seed();
