// Callers of every TypeScript function, by the TypeScript compiler's own
// reference search (the engine behind tsserver's "Find All References").
//
//   node eval/ts_gold.cjs <repo root> <typescript module dir>  < files.json
//
// stdin: a JSON list of the repo-relative files SEI indexed. stdout: a JSON list
// of {rel_path, qualified_name, line, column, callers}. Run by eval/benchmark.py;
// `typescript` is a dev-only tool installed outside the repository (pinned there).
//
// Functions are enumerated and named the way adapters/typescript.py names them,
// so the two sides compare by qualified name: function declarations (overload
// signatures too), classes and their methods, `const f = () => …` bindings, and
// an anonymous default export as `<module>.default`; bodies of functions and
// methods are descended into, arrow bodies are not. A reference counts when it
// is the callee of a call — `f(…)` or `x.f(…)` — not an import, a mention or a
// JSX element (SEI has no edge for rendering either). It is attributed to the
// nearest enclosing symbol, as SEI's `_owner_of` does; a call at module level
// has none and is skipped.
"use strict";

const fs = require("fs");
const path = require("path");

const root = path.resolve(process.argv[2]);
const ts = require(path.resolve(process.argv[3]));
const files = JSON.parse(fs.readFileSync(0, "utf8"));
const absolute = (rel) => path.join(root, rel).split(path.sep).join("/");
const relOf = new Map(files.map((rel) => [absolute(rel), rel]));

function moduleName(rel) {
  let p = rel;
  for (const suffix of [".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs"]) {
    if (p.endsWith(suffix)) { p = p.slice(0, -suffix.length); break; }
  }
  return p.split("/").filter(Boolean).join(".");
}

// -- the language service over the indexed files, with the repo's tsconfig ----

let options = { allowJs: true, jsx: ts.JsxEmit.Preserve, noEmit: true };
const configPath = path.join(root, "tsconfig.json");
if (fs.existsSync(configPath)) {
  const read = ts.readConfigFile(configPath, ts.sys.readFile);
  if (!read.error) {
    options = { ...ts.parseJsonConfigFileContent(read.config, ts.sys, root).options, noEmit: true };
  }
}
const texts = new Map();
const host = {
  getScriptFileNames: () => [...relOf.keys()],
  getScriptVersion: () => "1",
  getScriptSnapshot: (name) => {
    if (!texts.has(name)) {
      if (!fs.existsSync(name)) return undefined;
      texts.set(name, fs.readFileSync(name, "utf8"));
    }
    return ts.ScriptSnapshot.fromString(texts.get(name));
  },
  getCurrentDirectory: () => root,
  getCompilationSettings: () => options,
  getDefaultLibFileName: (o) => ts.getDefaultLibFilePath(o),
  fileExists: ts.sys.fileExists,
  readFile: ts.sys.readFile,
  readDirectory: ts.sys.readDirectory,
  directoryExists: ts.sys.directoryExists,
  getDirectories: ts.sys.getDirectories,
};
const service = ts.createLanguageService(host, ts.createDocumentRegistry());
const program = service.getProgram();

// -- functions, named as SEI names them ----------------------------------------

const functions = [];            // {rel, qn, nameNode, file}
const owners = new Map();        // node -> qualified name, for attribution

const isExported = (node) => (ts.getCombinedModifierFlags(node) & ts.ModifierFlags.Export) !== 0;
const isDefault = (node) => (ts.getCombinedModifierFlags(node) & ts.ModifierFlags.Default) !== 0;

function addFunction(file, rel, node, qn, nameNode) {
  owners.set(node, qn);
  functions.push({ rel, qn, nameNode: nameNode || node, file });
}

function collect(file, rel, statements, scope) {
  for (const node of statements) {
    if (ts.isFunctionDeclaration(node)) {
      const name = node.name ? node.name.text : (isExported(node) && isDefault(node) ? "default" : null);
      if (!name) continue;
      const qn = scope ? `${scope}.${name}` : name;
      addFunction(file, rel, node, qn, node.name);
      if (node.body) collect(file, rel, node.body.statements, qn);
    } else if (ts.isClassDeclaration(node) && node.name) {
      const qn = scope ? `${scope}.${node.name.text}` : node.name.text;
      owners.set(node, qn);
      for (const member of node.members) {
        const isMethod = ts.isMethodDeclaration(member) || ts.isConstructorDeclaration(member)
          || ts.isGetAccessorDeclaration(member) || ts.isSetAccessorDeclaration(member);
        if (!isMethod || !member.body) continue;
        const name = ts.isConstructorDeclaration(member) ? "constructor"
          : (member.name && (ts.isIdentifier(member.name) || ts.isPrivateIdentifier(member.name))
            ? member.name.text : null);
        if (!name) continue;
        const mqn = `${qn}.${name}`;
        addFunction(file, rel, member, mqn, member.name || member);
        collect(file, rel, member.body.statements, mqn);
      }
    } else if (ts.isVariableStatement(node)) {
      for (const decl of node.declarationList.declarations) {
        const value = decl.initializer;
        if (!value || !ts.isIdentifier(decl.name)) continue;
        if (!ts.isArrowFunction(value) && !ts.isFunctionExpression(value)) continue;
        const qn = scope ? `${scope}.${decl.name.text}` : decl.name.text;
        addFunction(file, rel, value, qn, decl.name);       // arrow bodies: not descended
      }
    } else if (ts.isExportAssignment(node) && !node.isExportEquals
               && (ts.isArrowFunction(node.expression) || ts.isFunctionExpression(node.expression))) {
      const qn = scope ? `${scope}.default` : "default";
      addFunction(file, rel, node.expression, qn, node.expression);
    } else if (ts.isBlock(node)) {
      collect(file, rel, node.statements, scope);
    } else if (ts.isIfStatement(node)) {
      collect(file, rel, [node.thenStatement, ...(node.elseStatement ? [node.elseStatement] : [])], scope);
    } else if (ts.isTryStatement(node)) {
      collect(file, rel, [node.tryBlock, ...(node.catchClause ? [node.catchClause.block] : []),
        ...(node.finallyBlock ? [node.finallyBlock] : [])], scope);
    } else if (ts.isSwitchStatement(node)) {
      for (const clause of node.caseBlock.clauses) collect(file, rel, clause.statements, scope);
    } else if (ts.isForStatement(node) || ts.isWhileStatement(node)) {
      collect(file, rel, [node.statement], scope);
    }
  }
}

for (const [abs, rel] of relOf) {
  const file = program.getSourceFile(abs);
  if (file) collect(file, rel, file.statements, moduleName(rel));
}

// -- callers ---------------------------------------------------------------------

function nodeAt(file, pos) {
  let found = file;
  const visit = (node) => {
    if (pos >= node.getStart(file) && pos < node.getEnd()) {
      found = node;
      ts.forEachChild(node, visit);
    }
  };
  ts.forEachChild(file, visit);
  return found;
}

function isCallee(id) {
  const parent = id.parent;
  if (!parent) return false;
  if (ts.isCallExpression(parent) && parent.expression === id) return true;
  return ts.isPropertyAccessExpression(parent) && parent.name === id
    && parent.parent && ts.isCallExpression(parent.parent) && parent.parent.expression === parent;
}

function ownerOf(node) {
  for (let cur = node.parent; cur; cur = cur.parent) {
    if (owners.has(cur)) return owners.get(cur);
  }
  return null;
}

const out = [];
for (const fn of functions) {
  const at = fn.nameNode.getStart(fn.file);
  const { line, character } = fn.file.getLineAndCharacterOfPosition(at);
  const callers = new Set();
  let refs = [];
  try {
    refs = service.findReferences(fn.file.fileName, at) || [];
  } catch (e) {
    refs = [];
  }
  for (const group of refs) {
    for (const ref of group.references) {
      if (ref.isDefinition) continue;
      const name = ref.fileName.split(path.sep).join("/");
      if (!relOf.has(name)) continue;
      const file = program.getSourceFile(name);
      if (!file) continue;
      const id = nodeAt(file, ref.textSpan.start);
      if (!ts.isIdentifier(id) && !ts.isPrivateIdentifier(id)) continue;
      if (!isCallee(id)) continue;
      const owner = ownerOf(id);
      if (owner) callers.add(owner);
    }
  }
  out.push({ rel_path: fn.rel, qualified_name: fn.qn, line: line + 1, column: character,
             callers: [...callers].sort() });
}
process.stdout.write(JSON.stringify(out));
