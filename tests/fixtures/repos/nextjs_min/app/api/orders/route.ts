import { NextRequest, NextResponse } from 'next/server';
import { listOrders, getOrderById } from '@/lib/orders';
import type { Order } from '@/lib/types';

/**
 * A second App Router handler exported under the same bare name as
 * app/api/users/route.ts. This is the collision §3.2 and §5.1 both describe:
 * every route file in the tree exports GET, so a stack frame carrying the bare
 * name `GET` cannot identify which one it came from.
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  const id = request.nextUrl.searchParams.get('id');
  if (id) {
    const order: Order | null = await getOrderById(id);
    return NextResponse.json(order ?? {}, { status: order ? 200 : 404 });
  }
  return NextResponse.json(await listOrders());
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  return NextResponse.json({ created: body }, { status: 201 });
}
