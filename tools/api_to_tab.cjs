// api_to_tab.cjs api.json "Folder/Name.json" note.md [stages|flow] [queue] : import an API graph into the real ComfyUI frontend, lay it out in
// columns by graph depth (guides grouped), add a how-to note, and save it as a workflow tab via /api/userdata.
const pw = require(process.env.PW || 'playwright-core'); const fs = require('fs');
const HOST = process.env.COMFY || 'http://127.0.0.1:8188';
(async () => {
  const api = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')); const dest = process.argv[3]; const note = fs.readFileSync(process.argv[4], 'utf8'); const layout = process.argv[5] || 'stages'; const doQueue = process.argv[6] === 'queue'; const front = process.env.FRONT === '1';   // FRONT=1: run next, ahead of the queue
  const b = await pw.chromium.launch({ headless: true }); const p = await b.newPage({ viewport: { width: 1700, height: 1000 } });
  p.on('pageerror', e => console.log('PAGEERR', String(e).slice(0, 200)));
  await p.goto(HOST, { waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 }); await p.waitForTimeout(4000);
  const res = await p.evaluate(async ({ api, dest, note, layout, doQueue, front }) => {
    const app = window.app, LG = window.LiteGraph;
    await app.loadGraphData({ nodes: [], links: [], version: 0.4, last_node_id: 0, last_link_id: 0, groups: [], config: {}, extra: {} });
    await app.loadApiJson(JSON.parse(JSON.stringify(api)), dest.split('/').pop().replace('.json', ''));
    await new Promise(r => setTimeout(r, 3000));
    const g = app.graph; const byId = {};
    for (const n of g._nodes) { byId[n.id] = n; const m = api[String(n.id)]; if (m && m._meta && m._meta.title) n.title = m._meta.title; }
    if (layout === 'flow') {   // small pipeline graphs: one column per graph depth, left to right
      const depth = {}; const d = id => { if (depth[id] !== undefined) return depth[id]; depth[id] = 0; const m = api[String(id)]; let best = 0;
        if (m) for (const v of Object.values(m.inputs)) if (Array.isArray(v) && api[String(v[0])]) best = Math.max(best, d(v[0]) + 1); return (depth[id] = best); };
      const cols = {}; for (const n of g._nodes) (cols[d(n.id)] = cols[d(n.id)] || []).push(n);
      let x = 0; for (const k of Object.keys(cols).map(Number).sort((a, b) => a - b)) { let y = 0, w = 0;
        for (const n of cols[k].sort((a, b) => a.id - b.id)) { n.size = [Math.max(n.size[0], 380), n.size[1]]; n.pos = [x, y]; y += n.size[1] + 80; w = Math.max(w, n.size[0]); }
        x += w + 100; }
    } else {
    // stage groups in builder order (ltx_graph adds nodes sequentially): models, inputs, prompt, stage 1, stage 2, output
    const STAGES = [['1 MODELS', null], ['2 INPUTS: identity anchors + depth control', ['LoadImage', 'VHS_LoadVideo']], ['3 PROMPT', ['CLIPTextEncode']],
      ['4 STAGE 1 (half res): guides + sample', ['ImageScale', 'EmptyLTXVLatentVideo', 'LTXVAddGuide']], ['5 STAGE 2 (full res): upscale, guides, refine', ['LTXVLatentUpsampler']],
      ['6 OUTPUT', ['VAEDecodeTiled']]];
    let si = 0; const groups = STAGES.map(() => []);
    for (const id of Object.keys(api).map(Number).sort((a, b) => a - b)) {
      const n = byId[id]; if (!n) continue; const cls = api[String(id)].class_type;
      for (let j = si + 1; j < STAGES.length; j++) if (STAGES[j][1].includes(cls)) { si = j; break; }
      groups[si].push(n);
    }
    const COLH = 1300, PAD = 40, TITLE = 50; let x = 0;
    groups.forEach((nodes, gi) => {
      if (!nodes.length) return;
      let cx = x + PAD, cy = TITLE + PAD, colW = 0, maxY = cy;
      for (const n of nodes) {
        n.size = [Math.max(n.size[0], 320), n.size[1]];
        if (cy + n.size[1] > COLH && cy > TITLE + PAD) { cx += colW + 40; cy = TITLE + PAD; colW = 0; }
        n.pos = [cx, cy]; cy += n.size[1] + 50; colW = Math.max(colW, n.size[0]); maxY = Math.max(maxY, cy);
      }
      const grp = new LG.LGraphGroup(STAGES[gi][0]); grp.pos = [x, 0]; grp.size = [cx + colW + PAD - x, Math.max(maxY, COLH) + PAD];
      grp.color = gi === 1 ? '#a1309b' : '#3f789e'; g.add(grp); x += grp.size[0] + 100;
    });
    }
    const nt = LG.createNode('MarkdownNote'); nt.pos = [0, -760]; nt.size = [1100, 700]; nt.title = 'HOW TO USE'; nt.widgets[0].value = note; g.add(nt);
    app.graph.setDirtyCanvas(true, true);
    const wf = app.graph.serialize();
    const r = await fetch('/api/userdata/' + encodeURIComponent('workflows/' + dest) + '?overwrite=true', { method: 'POST', body: JSON.stringify(wf) });
    let prompt_id = null;
    if (doQueue) {   // queue exactly like pressing Run, so the workflow travels with the prompt and shows in the queue/history panel
      const pr = await app.graphToPrompt();
      const resp = await fetch('/prompt', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: pr.output, ...(front ? { front: true } : {}), extra_data: { extra_pnginfo: { workflow: pr.workflow } } }) });   // no client_id: ComfyUI broadcasts progress to every open tab
      const body = await resp.json(); prompt_id = body.prompt_id || null; if (!prompt_id) return { error: body };
    }
    return { nodes: g._nodes.length, links: wf.links.length, save: r.status, prompt_id };
  }, { api, dest, note, layout, doQueue, front });
  console.log(JSON.stringify(res)); await b.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });
