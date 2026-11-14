type User = { id: string; name: string }

export async function GET() {
  const users = await loadUsers()
  return users.map((user) => user.id)
}

export async function POST(body: { name?: string }) {
  if (!body.name) {
    throw new Error("bad request")
  }
  return { ok: true }
}

async function loadUsers(): Promise<User[]> {
  return []
}
