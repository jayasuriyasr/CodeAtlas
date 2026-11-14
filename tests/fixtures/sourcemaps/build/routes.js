define("app/api/users/route", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.GET = GET;
    exports.POST = POST;
    async function GET() {
        const users = await loadUsers();
        return users.map((user) => user.id);
    }
    async function POST(body) {
        if (!body.name) {
            throw new Error("bad request");
        }
        return { ok: true };
    }
    async function loadUsers() {
        return [];
    }
});
define("app/api/orders/route", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.GET = GET;
    exports.POST = POST;
    async function GET() {
        const orders = await loadOrders();
        return orders.filter((order) => order.total > 0);
    }
    async function POST(body) {
        if (body.total === undefined) {
            throw new Error("missing total");
        }
        return { ok: true };
    }
    async function loadOrders() {
        return [];
    }
});
//# sourceMappingURL=routes.js.map