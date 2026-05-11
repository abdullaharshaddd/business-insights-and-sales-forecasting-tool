const { PrismaClient } = require('@prisma/client');
const fs = require('fs');
const path = require('path');

const prisma = new PrismaClient();

async function restoreOperationalData() {
    console.log("Cleaning up Olist operational data from PostgreSQL...");
    
    const tables = [
      'order_reviews', 'order_items', 'purchase_order_items', 'purchase_orders',
      'inventory', 'stock_movements', 'products', 'categories', 'suppliers'
    ];
    
    for (const table of tables) {
      try {
        await prisma.$executeRawUnsafe(`TRUNCATE TABLE "${table}" CASCADE;`);
      } catch (e) {
        // Table might not exist or already be empty
      }
    }

    console.log("Loading Online Retail products for Inventory Management...");
    
    const supplier = await prisma.supplier.create({
        data: {
            name: 'Global Retail Supplies',
            contactPerson: 'John Stock',
            email: 'supplies@globalretail.com',
            phone: '+1-555-0199',
            isActive: true
        }
    });

    const category = await prisma.category.create({
        data: {
            name: 'General Merchandise',
            slug: 'general-merchandise',
            namePt: 'Mercadoria Geral',
            description: 'Online Retail items'
        }
    });

    const jsonPath = path.join(__dirname, '../../products_temp.json');
    const products = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
    console.log(`Found ${products.length} products in JSON.`);

    let count = 0;
    for (const p of products) {
        try {
            const sku = p.sku;
            await prisma.product.create({
                data: {
                    sku: sku,
                    name: p.name.toLowerCase().split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
                    description: `Online Retail item: ${p.name}`,
                    basePrice: p.price,
                    reorderPoint: 20,
                    reorderQty: 50,
                    categoryId: category.id,
                    supplierId: supplier.id,
                    status: 'active',
                    inventory: {
                        create: {
                            quantity: Math.floor(Math.random() * 150) + 5,
                            reservedQty: 0
                        }
                    }
                }
            });
            count++;
            if (count % 100 === 0) console.log(`Migrated ${count} products...`);
        } catch (e) {
            console.error(`Error creating product ${sku}:`, e.message);
        }
    }

    console.log(`Successfully restored ${count} human-readable products.`);
}

restoreOperationalData()
    .catch(console.error)
    .finally(() => prisma.$disconnect());
