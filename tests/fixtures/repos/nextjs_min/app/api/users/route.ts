import { NextRequest, NextResponse } from 'next/server';
import { getUserById, listUsers } from '@/lib/users';
import type { User } from '@/lib/types';

export const dynamic = 'force-dynamic';

/**
 * App Router route handler. Every route.ts in the tree exports a function
 * named GET, which is exactly the bare-name collision spec §3.2 and §5.1
 * describe: the name alone cannot identify the symbol.
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  const id = request.nextUrl.searchParams.get('id');
  if (id) {
    const user: User | null = await getUserById(id);
    return NextResponse.json(user ?? {}, { status: user ? 200 : 404 });
  }
  return NextResponse.json(await listUsers());
}

export async function POST(request: NextRequest, context?: { params: unknown }) {
  const body = await request.json();
  return NextResponse.json({ created: body }, { status: 201 });
}

// Overloads: three declarations, one qualified_name, one arity. Without the
// source-order ordinal in the UID (§3.2 T1) these collapse to one node.
export function parse(x: string): User;
export function parse(x: number): User;
export function parse(x: any): User {
  return { id: String(x), name: '' } as User;
}

function normalize(...parts: string[]): string {
  return parts.join('/');
}

const withDefault = (limit: number = 20, cursor?: string) => ({ limit, cursor });
