import { execSync, spawnSync } from 'child_process';
import os from 'os';
import fs from 'fs';
const tgzDir='/tmp/tgzs', metaDir='/tmp/meta';
fs.mkdirSync(tgzDir,{recursive:true}); fs.mkdirSync(metaDir,{recursive:true});
const ARCH = `${os.platform()}-${os.arch()}`; // e.g. linux-arm64
const isPlatformBinding = n => /^@(esbuild|rollup|lightningcss)\//.test(n);
function keepOptional(name){
  // only install platform binding for the current arch+platform
  if(!isPlatformBinding(name)) return true;
  const base = name.replace(/^@[^/]+\//,'').replace(/^rollup-?/,'');
  // keep if it mentions linux + arm64 (e.g. rollup-linux-arm64-gnu)
  return name.includes('linux') && (name.includes('arm64'));
}
function curl(url, out){
  for(let i=0;i<5;i++){
    try{
      execSync(`curl -sfL --retry 8 --retry-all-errors --retry-delay 1 --max-time 300 "${url}" -o "${out}"`,{timeout:320});
      if(fs.existsSync(out)&&fs.statSync(out).size>0) return true;
    }catch(e){}
  }
  return false;
}
function meta(name, range){
  const f=`${metaDir}/${name.replace(/\//g,'_').replace(/@/g,'at')}@${range}.json`;
  if(fs.existsSync(f)) return JSON.parse(fs.readFileSync(f));
  for(let i=0;i<3;i++){
    try{
      const r=execSync(`npm view "${name}@${range}" --json`,{stdio:['ignore','pipe','ignore'],maxBuffer:2e8,timeout:40000}).toString();
      let m=JSON.parse(r); if(Array.isArray(m))m=m[0];
      fs.writeFileSync(f, JSON.stringify(m));
      return m;
    }catch(e){}
  }
  throw new Error('meta fail '+name+'@'+range);
}
// resolve tree
const rootDeps=JSON.parse(process.argv[2]);
const chosen=new Map();
const queue=[...Object.entries(rootDeps)];
let guard=0, skipTypes=0;
while(queue.length&&guard++<400){
  const [name,range]=queue.shift();
  if(chosen.has(name))continue;
  // type-only packages not needed at runtime
  if(name.startsWith('@types/')){
    try{ meta(name,range); chosen.set(name,meta(name,range).version||'skip'); }catch(e){ skipTypes++; }
    continue;
  }
  try{
    const m=meta(name,range);
    const ver=m.version||(m['dist-tags']&&m['dist-tags'].latest);
    if(!ver){console.error('no ver',name,range);continue;}
    chosen.set(name,ver);
    const deps={...m.dependencies||{}};
    for(const [dn,dr] of Object.entries(m.optionalDependencies||{})) if(keepOptional(dn)) deps[dn]=dr;
    for(const [dn,dr] of Object.entries(deps)) if(!chosen.has(dn)&&!queue.some(q=>q[0]===dn)) queue.push([dn,dr]);
  }catch(e){console.error('ERR',name,e.message);}
}
console.log('CHOSEN', chosen.size, 'skipTypes', skipTypes);
const names=[...chosen.keys()].sort();
// download + extract
const nm='/tmp/pp-test/node_modules';
fs.mkdirSync(nm,{recursive:true});
let dl=0;
for(const name of names){
  const ver=chosen.get(name);
  if(ver==='skip') continue;
  let tb=null;
  try{
    const m=meta(name,ver);
    tb=m.dist&&m.dist.tarball;
  }catch(e){}
  if(!tb){ console.error('no tarball',name); continue; }
  const tgz=`${tgzDir}/${name.replace(/\//g,'_').replace(/@/g,'at')}@${ver}.tgz`;
  if(!curl(tb,tgz)){console.error('DL FAIL',name,ver);continue;}
  const target = name.startsWith('@') ? `${nm}/${name.split('/')[0]}/${name.split('/')[1]}` : `${nm}/${name}`;
  fs.rmSync(target,{recursive:true,force:true});
  fs.mkdirSync(target,{recursive:true});
  const r=spawnSync('tar',['-xzf',tgz,'--strip-components=1','-C',target],{stdio:'ignore'});
  if(r.status!==0){console.error('TAR FAIL',name);continue;}
  dl++;
}
console.log('DOWNLOADED',dl,'of',names.length);
