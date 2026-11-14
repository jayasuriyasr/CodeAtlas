type Order = { id: string; total: number }

export async function GET() {
  const orders = await loadOrders()
  return orders.filter((order) => order.total > 0)
}

export async function POST(body: { total?: number }) {
  if (body.total === undefined) {
    throw new Error("missing total")
  }
  return { ok: true }
}

async function loadOrders(): Promise<Order[]> {
  return []
}
