// tools/audit.mjs — runtime contract audit. Connects to the live app and checks
// that every module registered, exposes its public API, and wired core accessors.
//
//   node tools/audit.mjs            # default http://127.0.0.1:5173/
import { launchChrome } from './cdp.mjs';

export async function audit(url = 'http://127.0.0.1:5173/') {
  const client = await launchChrome({ url });
  try {
    await client.waitForReady();
    await new Promise(r => setTimeout(r, 1200));
    const expr = `(()=>{
      const w=window.__skylines.world;
      const ids=['terrain','environment','roads','simulation','effects','zoning','buildings','props','traffic','tools','audio','ui','demo'];
      const out={modules:{},dead:[...w.dead]};
      for(const id of ids){
        const m=w.modules[id];
        out.modules[id]=m?{registered:true,init:typeof m.init==='function',update:typeof m.update==='function',showcase:typeof m.showcase==='function'}:{registered:false};
      }
      out.terrainHeightAt=typeof w.terrain?.heightAt==='function';
      out.terrainNormalAt=typeof w.terrain?.normalAt==='function';
      out.simState=w.simulation?JSON.stringify(w.simulation):null;
      out.tick=w.meta.tick;
      return JSON.stringify(out);
    })()`;
    const res = await client.evalJS(expr);
    return JSON.parse(res);
  } finally { client.close(); }
}

if (import.meta.url === new URL(process.argv[1], 'file:').href) {
  audit().then(a => {
    for (const [id, info] of Object.entries(a.modules)) {
      console.log(`${id.padEnd(12)} registered=${info.registered} init=${info.init} update=${info.update} showcase=${info.showcase}`);
    }
    console.log(`terrain.heightAt=${a.terrainHeightAt} normalAt=${a.terrainNormalAt}`);
    console.log(`sim=${a.simState}`);
    console.log(`tick=${a.tick} dead=[${a.dead}]`);
  }).catch(e => { console.error('audit failed:', e.message); process.exit(1); });
}
