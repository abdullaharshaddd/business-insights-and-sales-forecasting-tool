# BISFT Database Architecture

This document provides a comprehensive overview of the storage architecture for the **Business Insights & Sales Forecasting Tool (BISFT)**. 

The system has transitioned from a fragmented SQLite/CSV model to a **Unified Production-Grade Architecture** powered primarily by **PostgreSQL**.

---

## 1. Primary Store: PostgreSQL (`bisft_inventory`)
The central source of truth for the entire application. Both the **Node.js Inventory Backend** and the **Python Analytics Engine** connect to this single instance.

### **A. Operational Schema (Inventory Management)**
These tables power the real-time business operations.
- **`users`**: Authentication and RBAC (Admin, Manager, Staff).
- **`suppliers`**: Supplier contact info, lead times, and terms.
- **`products`**: The master product catalog (Unified with Olist data).
- **`inventory`**: Current stock levels (`quantity`), reserved stock, and restock dates.
- **`stock_movements`**: Immutable audit trail of every item added or removed (In/Out/Adjust).
- **`purchase_orders`**: Procurement tracking from draft to received.
- **`categories`**: Hierarchical product categorization.

### **B. Historical Analytics Schema (Olist Marketplace)**
Migrated from the original Olist SQLite dataset into PostgreSQL for high-performance BI queries.
- **`orders`**: Transactional history (99k+ records).
- **`customers`**: Demographic data and unique identifiers.
- **`order_items`**: Junction table linking orders, products, and sellers (suppliers).
- **`order_payments`**: Installment and payment type data.
- **`order_reviews`**: Customer satisfaction scores (1-5 stars).
- **`geolocation`**: 360k+ coordinate points for geographic distribution analysis.

---

## 2. AI Intelligence Storage

### **A. Vector Database (ChromaDB)**
- **Location**: `data/vector_db/`
- **Purpose**: Powering the **RAG (Retrieval-Augmented Generation)** system.
- **Content**: Contains embedded chunks of business strategy playbooks, SOPs, and market reports.
- **Used By**: The `Chat` router and `Data Gatherer` agent to provide context-aware consulting.

### **B. Long-Term Analytical Memory (PostgreSQL)**
Stored in the main PostgreSQL database but used exclusively by the AI agents.
- **`findings`**: Stores deterministic results from the KPI Engine (e.g., "Revenue dropped by 5% in SP").
- **`conversation_summaries`**: Stores thread-based summaries to maintain context across long chat sessions.

---

## 3. Legacy / Development Storage

### **Olist SQLite (`olist.db`)**
- **Location**: `data/processed/olist/olist.db`
- **Status**: **DEPRECATED**
- **Role**: This is now only used as a reference source. The data has been fully migrated to PostgreSQL using the `inventory-backend/src/seed/seed-from-olist.ts` script.

---

## 4. Connection Configuration

To maintain consistency across the stack, the following connection logic is enforced:

### **Node.js (Prisma)**
Configured in `inventory-backend/.env`:
`DATABASE_URL="postgresql://user:pass@localhost:5432/bisft_inventory?schema=public"`

### **Python (SQLAlchemy)**
Configured in `config/config.yaml`:
```yaml
database:
  url: "postgresql://user:pass@localhost:5432/bisft_inventory"
```

---

## 5. Entity-Relationship Highlights

1.  **Unified Products**: The `products` table in the Inventory module uses the same UUIDs as the Olist `product_id`. This allows us to join historical sales trends directly with current stock levels.
2.  **Sellers as Suppliers**: Olist `sellers` are mapped to the `suppliers` table, enabling the "Purchase Order" system to restock products from the same entities that sold them historically.
3.  **Audit Integrity**: Every change in the `inventory` table triggers an entry in `stock_movements` and `audit_logs`, ensuring a full production-grade audit trail.

---

## 6. Maintenance Commands

- **Update Schema**: `npx prisma migrate dev` (inside `inventory-backend`)
- **Regenerate Client**: `npx prisma generate`
- **Re-Seed Operations**: `npm run seed` (to populate initial inventory levels)
