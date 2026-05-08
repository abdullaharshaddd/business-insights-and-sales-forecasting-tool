import app from './app';
import { env } from './config/env';
import prisma from './config/database';

async function main() {
  try {
    // Test database connection
    await prisma.$connect();
    console.log('✅ Database connected');

    app.listen(env.PORT, () => {
      console.log(`\n${'='.repeat(55)}`);
      console.log(`  BISFT Inventory Backend`);
      console.log(`  Environment: ${env.NODE_ENV}`);
      console.log(`  Port: ${env.PORT}`);
      console.log(`  API: http://localhost:${env.PORT}/api/v1`);
      console.log(`${'='.repeat(55)}\n`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\nShutting down...');
  await prisma.$disconnect();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await prisma.$disconnect();
  process.exit(0);
});

main();
