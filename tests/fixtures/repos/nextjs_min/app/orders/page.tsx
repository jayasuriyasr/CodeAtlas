import { Suspense } from 'react';
import OrderList from '@/components/OrderList';
import type { Order } from '@/lib/types';

/**
 * An App Router page. §3.5's `(Route)-[:RENDERS]->(Component)` is this
 * relationship: the route renders its default export, which is a component —
 * not a request handler, which is what route.ts exports.
 */
export default async function OrdersPage({ searchParams }: { searchParams: Q }) {
  const orders: Order[] = await fetchOrders(searchParams);
  return (
    <Suspense fallback={<span>loading</span>}>
      <OrderList orders={orders} />
    </Suspense>
  );
}

export function generateMetadata() {
  return { title: 'Orders' };
}
