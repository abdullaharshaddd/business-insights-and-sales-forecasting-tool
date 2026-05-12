/**
 * seed-from-olist.ts
 * ==================
 * Seeds the PostgreSQL database with historical data from CSV files.
 *
 * Sources:
 *   - Olist data:    data/raw/olist/*.csv
 *   - Online Retail: data/processed/online_retail_cleaned.csv
 *
 * Tables seeded:
 *   1. Admin user
 *   2. Olist category translations
 *   3. Olist sellers
 *   4. Olist customers
 *   5. Olist products
 *   6. Olist orders
 *   7. Olist order items
 *   8. Olist order payments
 *   9. Olist order reviews
 *  10. Olist geolocation
 *  11. Online Retail customers
 *  12. Online Retail products
 *  13. Online Retail invoices
 *  14. Online Retail invoice items
 *
 * NOTE: Operational tables (products, inventory, suppliers, purchase_orders)
 * are for live inventory management — seed them via the UI, not here.
 */

import path from 'path';
import fs from 'fs';
import bcrypt from 'bcryptjs';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const DATA_DIR = path.resolve(__dirname, '../../../data');
const OLIST_DIR = path.join(DATA_DIR, 'raw/olist');
const RETAIL_CSV = path.join(DATA_DIR, 'processed/online_retail_cleaned.csv');
const BATCH_SIZE = 500;

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

function csvRows(filepath: string): any[] {
  const content = fs.readFileSync(filepath, 'utf-8');
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim());
    const row: any = {};
    headers.forEach((h, i) => { row[h] = values[i] ?? null; });
    return row;
  });
}

async function batchUpsert(tx: any, modelName: string, data: any[], uniqueField: string) {
  for (const item of data) {
    await (tx as any)[modelName].upsert({
      where: { [uniqueField]: item[uniqueField] },
      update: {},
      create: item,
    }).catch(() => { /* skip duplicates silently */ });
  }
}

async function seed() {
  console.log('='.repeat(55));
  console.log('  BISFT — Seed Historical Data');
  console.log('='.repeat(55));

  try {
    // 1. Admin User
    console.log('\n[1/14] Creating admin user...');
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

    // 2. Category Translations
    console.log('\n[2/14] Seeding Olist category translations...');
    const catRows = csvRows(path.join(OLIST_DIR, 'product_category_name_translation.csv'));
    for (const row of catRows) {
      await prisma.olistCategoryTranslation.upsert({
        where: { categoryNamePt: row.product_category_name },
        update: {},
        create: {
          categoryNamePt: row.product_category_name,
          categoryNameEn: row.product_category_name_english || row.product_category_name,
        },
      });
    }

    // 3. Olist Sellers
    console.log('\n[3/14] Seeding Olist sellers...');
    const sellers = csvRows(path.join(OLIST_DIR, 'olist_sellers_dataset.csv')).map((r: any) => ({
      sellerId: r.seller_id,
      zipCodePrefix: r.seller_zip_code_prefix ? String(r.seller_zip_code_prefix) : null,
      city: r.seller_city,
      state: r.seller_state,
    }));
    await batchUpsert(prisma, 'olistSeller', sellers, 'sellerId');
    console.log(`    Seeded ${sellers.length} sellers`);

    // 4. Olist Customers
    console.log('\n[4/14] Seeding Olist customers...');
    const customers = csvRows(path.join(OLIST_DIR, 'olist_customers_dataset.csv')).map((r: any) => ({
      customerId: r.customer_id,
      customerUniqueId: r.customer_unique_id,
      zipCodePrefix: r.customer_zip_code_prefix ? String(r.customer_zip_code_prefix) : null,
      city: r.customer_city,
      state: r.customer_state,
    }));
    await batchUpsert(prisma, 'olistCustomer', customers, 'customerId');
    console.log(`    Seeded ${customers.length} Olist customers`);

    // 5. Olist Products
    console.log('\n[5/14] Seeding Olist products...');
    const productRows = csvRows(path.join(OLIST_DIR, 'olist_products_dataset.csv')).map((r: any) => ({
      productId: r.product_id,
      categoryNamePt: r.product_category_name,
      nameLength: r.product_name_lenght ? parseInt(r.product_name_lenght) : null,
      descriptionLength: r.product_description_lenght ? parseInt(r.product_description_lenght) : null,
      photosQty: Math.round(parseFloat(r.product_photos_qty) || 0),
      weightG: r.product_weight_g ? Math.round(parseFloat(r.product_weight_g)) : null,
      lengthCm: r.product_length_cm ? parseFloat(r.product_length_cm) : null,
      heightCm: r.product_height_cm ? parseFloat(r.product_height_cm) : null,
      widthCm: r.product_width_cm ? parseFloat(r.product_width_cm) : null,
    }));
    await batchUpsert(prisma, 'olistProduct', productRows, 'productId');
    console.log(`    Seeded ${productRows.length} Olist products`);

    // 6. Olist Orders
    console.log('\n[6/14] Seeding Olist orders...');
    const orders = csvRows(path.join(OLIST_DIR, 'olist_orders_dataset.csv')).map((r: any) => ({
      orderId: r.order_id,
      customerId: r.customer_id,
      orderStatus: r.order_status,
      orderPurchaseTimestamp: r.order_purchase_timestamp ? new Date(r.order_purchase_timestamp) : null,
      orderApprovedAt: r.order_approved_at ? new Date(r.order_approved_at) : null,
      orderDeliveredCarrierDate: r.order_delivered_carrier_date ? new Date(r.order_delivered_carrier_date) : null,
      orderDeliveredCustomerDate: r.order_delivered_customer_date ? new Date(r.order_delivered_customer_date) : null,
      orderEstimatedDeliveryDate: r.order_estimated_delivery_date ? new Date(r.order_estimated_delivery_date) : null,
    }));
    await batchUpsert(prisma, 'olistOrder', orders, 'orderId');
    console.log(`    Seeded ${orders.length} Olist orders`);

    // 7. Olist Order Items
    console.log('\n[7/14] Seeding Olist order items...');
    const orderItems = csvRows(path.join(OLIST_DIR, 'olist_order_items_dataset.csv')).map((r: any) => ({
      orderId: r.order_id,
      orderItemId: parseInt(r.order_item_id),
      productId: r.product_id || null,
      sellerId: r.seller_id || null,
      shippingLimitDate: r.shipping_limit_date ? new Date(r.shipping_limit_date) : null,
      price: r.price ? parseFloat(r.price) : null,
      freightValue: r.freight_value ? parseFloat(r.freight_value) : null,
    }));
    await batchUpsert(prisma, 'olistOrderItem', orderItems, 'orderId_orderItemId');
    console.log(`    Seeded ${orderItems.length} Olist order items`);

    // 8. Olist Order Payments
    console.log('\n[8/14] Seeding Olist order payments...');
    const payments = csvRows(path.join(OLIST_DIR, 'olist_order_payments_dataset.csv')).map((r: any) => ({
      orderId: r.order_id,
      paymentSequential: parseInt(r.payment_sequential),
      paymentType: r.payment_type,
      paymentInstallments: r.payment_installments ? parseInt(r.payment_installments) : null,
      paymentValue: r.payment_value ? parseFloat(r.payment_value) : null,
    }));
    await batchUpsert(prisma, 'olistOrderPayment', payments, 'orderId_paymentSequential');
    console.log(`    Seeded ${payments.length} Olist payments`);

    // 9. Olist Order Reviews
    console.log('\n[9/14] Seeding Olist order reviews...');
    const reviews = csvRows(path.join(OLIST_DIR, 'olist_order_reviews_dataset.csv')).map((r: any) => ({
      reviewId: r.review_id,
      orderId: r.order_id,
      reviewScore: r.review_score ? parseInt(r.review_score) : null,
      reviewCommentTitle: r.review_comment_title || null,
      reviewCommentMessage: r.review_comment_message || null,
      reviewCreationDate: r.review_creation_date ? new Date(r.review_creation_date) : null,
      reviewAnswerTimestamp: r.review_answer_timestamp ? new Date(r.review_answer_timestamp) : null,
    }));
    await batchUpsert(prisma, 'olistOrderReview', reviews, 'reviewId');
    console.log(`    Seeded ${reviews.length} Olist reviews`);

    // 10. Olist Geolocation
    console.log('\n[10/14] Seeding Olist geolocation...');
    const geo = csvRows(path.join(OLIST_DIR, 'olist_geolocation_dataset.csv')).map((r: any) => ({
      zipCodePrefix: String(r.geolocation_zip_code_prefix),
      lat: r.geolocation_lat ? parseFloat(r.geolocation_lat) : null,
      lng: r.geolocation_lng ? parseFloat(r.geolocation_lng) : null,
      city: r.geolocation_city,
      state: r.geolocation_state,
    }));
    await batchUpsert(prisma, 'olistGeolocation', geo, 'id');
    console.log(`    Seeded ${geo.length} Olist geolocation records`);

    // 11. Online Retail Customers
    console.log('\n[11/14] Seeding Online Retail customers...');
    const retailRows = csvRows(RETAIL_CSV);
    const retailCustomersMap = new Map<string, string>(); // customerid -> country
    for (const r of retailRows) {
      if (r.customerid) retailCustomersMap.set(r.customerid, r.country);
    }
    const retailCustomers = Array.from(retailCustomersMap.entries()).map(([customerid, country]) => ({
      customerid,
      country: country || 'Unknown',
    }));
    await batchUpsert(prisma, 'retailCustomer', retailCustomers, 'customerid');
    console.log(`    Seeded ${retailCustomers.length} Online Retail customers`);

    // 12. Online Retail Products
    console.log('\n[12/14] Seeding Online Retail products...');
    const retailProductsMap = new Map<string, { description: string; latestUnitPrice: number }>();
    for (const r of retailRows) {
      if (r.stockcode) {
        retailProductsMap.set(r.stockcode, {
          description: r.description || '',
          latestUnitPrice: r.unitprice ? parseFloat(r.unitprice) : 0,
        });
      }
    }
    const retailProducts = Array.from(retailProductsMap.entries()).map(([stockcode, data]) => ({
      stockcode,
      description: data.description,
      latestUnitPrice: data.latestUnitPrice,
    }));
    await batchUpsert(prisma, 'retailProduct', retailProducts, 'stockcode');
    console.log(`    Seeded ${retailProducts.length} Online Retail products`);

    // 13. Online Retail Invoices
    console.log('\n[13/14] Seeding Online Retail invoices...');
    const invoicesMap = new Map<string, { customerid: string; invoicedate: Date; isCancellation: boolean }>();
    for (const r of retailRows) {
      if (r.invoiceno) {
        invoicesMap.set(r.invoiceno, {
          customerid: r.customerid || '',
          invoicedate: r.invoicedate ? new Date(r.invoicedate) : new Date(),
          isCancellation: r.invoiceno.toString().startsWith('C'),
        });
      }
    }
    const invoices = Array.from(invoicesMap.entries()).map(([invoiceno, data]) => ({
      invoiceno,
      customerid: data.customerid,
      invoicedate: data.invoicedate,
      isCancellation: data.isCancellation,
    }));
    await batchUpsert(prisma, 'retailInvoice', invoices, 'invoiceno');
    console.log(`    Seeded ${invoices.length} Online Retail invoices`);

    // 14. Online Retail Invoice Items
    console.log('\n[14/14] Seeding Online Retail invoice items...');
    const invoiceItems = retailRows
      .filter((r: any) => r.invoiceno && r.stockcode && !r.invoiceno.toString().startsWith('C'))
      .map((r: any) => ({
        invoiceno: r.invoiceno,
        stockcode: r.stockcode,
        quantity: parseInt(r.quantity) || 0,
        unitprice: parseFloat(r.unitprice) || 0,
        totalprice: parseFloat(r.totalprice) || 0,
      }));
    // Remove duplicates (same invoice+stockcode combination might appear from cleaning)
    const seen = new Set<string>();
    const uniqueInvoiceItems = invoiceItems.filter((item: any) => {
      const key = `${item.invoiceno}-${item.stockcode}-${item.quantity}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    // Use raw insert for speed (no unique key on this table)
    if (uniqueInvoiceItems.length > 0) {
      await prisma.retailInvoiceItem.createMany({ data: uniqueInvoiceItems, skipDuplicates: true });
    }
    console.log(`    Seeded ${uniqueInvoiceItems.length} Online Retail invoice items`);

    console.log('\n✅ BISFT historical data seeded successfully!');
    console.log('   Operational tables (products, inventory, suppliers, POs)');
    console.log('   are empty — use the UI to manage live operational data.\n');

  } catch (err) {
    console.error('\n❌ Seeding failed:', err);
  } finally {
    await prisma.$disconnect();
  }
}

seed();
