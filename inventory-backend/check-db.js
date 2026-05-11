const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function check() {
  const products = await prisma.product.count();
  const suppliers = await prisma.supplier.count();
  const inventory = await prisma.inventory.count();
  const orders = await prisma.order.count();
  
  console.log('--- Database Status ---');
  console.log('Products:', products);
  console.log('Suppliers:', suppliers);
  console.log('Inventory Records:', inventory);
  console.log('Orders:', orders);
  
  if (products > 0) {
    const sample = await prisma.product.findFirst({ include: { inventory: true } });
    console.log('\nSample Product:', JSON.stringify(sample, null, 2));
  }
  
  await prisma.$disconnect();
}

check();
