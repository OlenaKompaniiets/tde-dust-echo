'use strict';
const D=window.CASE_A,E=D.epochs,C=D.dense,$=id=>document.getElementById(id);
const palette={W1:'#1677a5',W2:'#bd5729'};
const hosts={W1:8.072315984025277,W2:5.215414582541471};
let hit=null,lcTransform=null;
function context(canvas,height){const w=Math.max(280,canvas.clientWidth),ratio=window.devicePixelRatio||1;canvas.width=w*ratio;canvas.height=height*ratio;canvas.style.height=height+'px';const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.font='13px system-ui';return [ctx,w,height];}
function line(ctx,x1,y1,x2,y2,color,width=1){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}
function drawLC(){const [g,w,h]=context($('lc'),470),bands=$('band').value==='both'?['W1','W2']:[$('band').value],excess=$('excess').checked;
 const left=62,right=w-22,top=27,bottom=292,rTop=330,rBottom=419;
 const xmin=Math.min(...C.mjd),xmax=Math.max(...C.mjd),x=t=>left+(t-xmin)/(xmax-xmin)*(right-left);
 let vals=[];for(const b of bands){const host=excess?hosts[b]:0;E[b+'_data_mJy'].forEach((v,i)=>vals.push(v-host+E[b+'_sigma_mJy'][i],v-host-E[b+'_sigma_mJy'][i]));C[b+'_model_mJy'].forEach(v=>vals.push(v-host));}
 const lo=Math.min(0,...vals)-1,hi=Math.max(...vals)*1.08,y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
 let maxR=4;for(const b of bands)maxR=Math.max(maxR,...E[b+'_residual_sigma'].map(Math.abs));maxR=Math.ceil(maxR);const ry=v=>(rTop+rBottom)/2-v/maxR*(rBottom-rTop)/2;
 g.fillStyle='#536777';g.textAlign='left';g.fillText(excess?'Excess flux (mJy)':'Total flux (mJy)',left,16);g.fillText('(data − model) / σ',left,321);
 for(let k=0;k<=4;k++){let v=lo+k*(hi-lo)/4;line(g,left,y(v),right,y(v),'#e3e9ee');g.textAlign='right';g.fillText(v.toFixed(0),left-9,y(v)+4);}
 for(let k=0;k<=4;k++){let t=xmin+k*(xmax-xmin)/4;g.textAlign='center';g.fillText(Math.round(t),x(t),442);line(g,x(t),rBottom,x(t),rBottom+5,'#526779');}
 for(const v of [-3,0,3]){g.setLineDash(v?[4,4]:[]);line(g,left,ry(v),right,ry(v),v?'#bac6ce':'#728393');g.setLineDash([]);g.textAlign='right';g.fillText(v,left-9,ry(v)+4);}
 line(g,left,top,left,bottom,'#526779');line(g,left,rTop,left,rBottom,'#526779');g.textAlign='center';g.fillText('Observer time (MJD)',(left+right)/2,464);
 for(const b of bands){const host=excess?hosts[b]:0;g.strokeStyle=palette[b];g.lineWidth=2;g.beginPath();C.mjd.forEach((t,i)=>{const xx=x(t),yy=y(C[b+'_model_mJy'][i]-host);i?g.lineTo(xx,yy):g.moveTo(xx,yy);});g.stroke();
 E.mjd.forEach((t,i)=>{const xx=x(t),v=E[b+'_data_mJy'][i]-host,err=E[b+'_sigma_mJy'][i];line(g,xx,y(v-err),xx,y(v+err),palette[b]);line(g,xx-3,y(v-err),xx+3,y(v-err),palette[b]);line(g,xx-3,y(v+err),xx+3,y(v+err),palette[b]);g.fillStyle=palette[b];for(const yy of [y(v),ry(E[b+'_residual_sigma'][i])]){g.beginPath();g.arc(xx,yy,3.5,0,Math.PI*2);g.fill();}});
 }
 if(hit!==null){g.setLineDash([3,4]);line(g,x(E.mjd[hit]),top,x(E.mjd[hit]),rBottom,'#758b9b');g.setLineDash([]);}
 lcTransform={x,xmin,xmax,left,right,bands};
}
function showEpoch(i){hit=i;const excess=$('excess').checked;let parts=[`MJD ${E.mjd[i].toFixed(2)}`];for(const b of lcTransform.bands){const host=excess?hosts[b]:0;parts.push(`${b}: data ${(E[b+'_data_mJy'][i]-host).toFixed(2)} ± ${E[b+'_sigma_mJy'][i].toFixed(2)} mJy; model ${(E[b+'_model_mJy'][i]-host).toFixed(2)} mJy; residual ${E[b+'_residual_sigma'][i].toFixed(2)}σ`);}$('lcInfo').textContent=parts.join(' | ');drawLC();}
$('lc').addEventListener('pointermove',ev=>{const xx=ev.clientX-$('lc').getBoundingClientRect().left;let i=0;E.mjd.forEach((_,j)=>{if(Math.abs(lcTransform.x(E.mjd[j])-xx)<Math.abs(lcTransform.x(E.mjd[i])-xx))i=j;});showEpoch(i);});
for(const id of ['band','excess'])$(id).addEventListener('change',()=>{drawLC();if(hit!==null)showEpoch(hit);});
$('resetLC').onclick=()=>{$('band').value='both';$('excess').checked=false;hit=null;$('lcInfo').textContent='Move over an epoch to inspect the data, error and prediction.';drawLC();};
$('epochTable').innerHTML='<table><caption>Observed total fluxes and adopted errors (mJy); residuals in σ</caption><thead><tr><th>MJD</th><th>W1 ± σ</th><th>W1 model</th><th>W1 residual</th><th>W2 ± σ</th><th>W2 model</th><th>W2 residual</th></tr></thead><tbody>'+E.mjd.map((t,i)=>'<tr><td>'+t.toFixed(2)+'</td>'+['W1','W2'].map(b=>'<td>'+E[b+'_data_mJy'][i].toFixed(3)+' ± '+E[b+'_sigma_mJy'][i].toFixed(3)+'</td><td>'+E[b+'_model_mJy'][i].toFixed(3)+'</td><td>'+E[b+'_residual_sigma'][i].toFixed(3)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
let yaw=-.95,pitch=.44,drag=null,playing=false,lastFrame=0;
const stops=[[201,225,239],[116,173,209],[104,85,162],[190,71,118],[238,133,60],[245,194,75],[255,241,168]];
function rgb(t){const u=Math.max(0,Math.min(1,t/D.displayMax))*(stops.length-1),i=Math.min(stops.length-2,Math.floor(u)),f=u-i;return stops[i].map((v,j)=>Math.round(v*(1-f)+stops[i+1][j]*f));}
function rotate(p){const xx=Math.cos(yaw)*p[0]-Math.sin(yaw)*p[1],a=Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1];return [xx,Math.cos(pitch)*p[2]-Math.sin(pitch)*a,Math.cos(pitch)*a+Math.sin(pitch)*p[2]];}
function drawDust(){const[g,w,h]=context($('dust'),480),idx=+$('time').value,temps=D.temperatures[$('grain').value][idx],region=$('region').value,R=region==='compact'?.5:2,scale=Math.min(w*.39,h*.39)/R,cx=w/2,cy=h/2+8;
 const project=p=>{const q=rotate(p);return [cx+q[0]*scale,cy-q[1]*scale,q[2]];};
 const inRegion=(r)=>region==='compact'?r<=.50001:region==='extended'?r>=.99999:true;
 const pts=[];let missing=0,max=0,min=Infinity;
 D.points.forEach((p,i)=>{const r=Math.hypot(...p);if(!inRegion(r)||($('cut').checked&&p[0]>0))return;const t=temps[i];if(t===null){missing++;return;}max=Math.max(max,t);min=Math.min(min,t);pts.push({q:project(p),t});});
 pts.sort((a,b)=>a.q[2]-b.q[2]);
 // Fixed geometric boundaries, not an inferred density surface.
 const radii=region==='compact'?[.25,.5]:region==='extended'?[1,2]:[.5,2],theta=D.summary.config.theta*Math.PI/180;
 for(const r of radii)for(const sign of [-1,1]){g.beginPath();let first=true;for(let k=0;k<=160;k++){const phi=k/160*2*Math.PI,p=[r*Math.sin(theta)*Math.cos(phi),r*Math.sin(theta)*Math.sin(phi),sign*r*Math.cos(theta)];if($('cut').checked&&p[0]>0){first=true;continue;}const q=project(p);if(first){g.moveTo(q[0],q[1]);first=false;}else g.lineTo(q[0],q[1]);}g.strokeStyle='#c1cdd5';g.lineWidth=.8;g.stroke();}
 for(const p of pts){const a=.025+.875*Math.pow(Math.min(p.t/800,1),1.8);g.fillStyle=`rgba(${rgb(p.t).join(',')},${a})`;g.beginPath();g.arc(p.q[0],p.q[1],region==='compact'?2.6:2.2,0,Math.PI*2);g.fill();}
 const origin=project([0,0,0]);line(g,origin[0]-4,origin[1],origin[0]+4,origin[1],'#283f50');line(g,origin[0],origin[1]-4,origin[0],origin[1]+4,'#283f50');
 g.fillStyle='#536777';g.textAlign='left';g.fillText(`Extent: r ≤ ${R} pc · drag to rotate`,14,21);g.fillText('Local rest-frame time; not observer time',14,h-12);
 const o=[w-69,h-67];for(let j=0;j<3;j++){const p=[0,0,0];p[j]=1;const q=rotate(p);line(g,...o,o[0]+q[0]*36,o[1]-q[1]*36,'#728798',1.3);g.fillText(['x','y','z'][j],o[0]+q[0]*44,o[1]-q[1]*44);}
 $('timeLabel').textContent=D.times[idx]+' d';$('dustInfo').textContent=`Displayed temperatures: ${pts.length?min.toFixed(1)+'–'+max.toFixed(1)+' K':'no supported samples'} · ${pts.length} valid samples · ${missing} outside stored temporal support. Colours share the fixed 0–1200 K scale.`;
}
$('dust').style.touchAction='none';$('dust').addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];$('dust').setPointerCapture(e.pointerId);});$('dust').addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.009;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];drawDust();});$('dust').addEventListener('pointerup',()=>drag=null);$('dust').addEventListener('pointercancel',()=>drag=null);
for(const id of ['time','region','grain','cut'])$(id).addEventListener('input',drawDust);
$('reset3D').onclick=()=>{yaw=-.95;pitch=.44;$('time').value=15;$('region').value='all';$('grain').value='mean';$('cut').checked=false;playing=false;$('play').textContent='▶ Play';drawDust();};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'▶ Play';};
function animate(now){if(playing&&now-lastFrame>170){$('time').value=(+$('time').value+1)%D.times.length;drawDust();lastFrame=now;}requestAnimationFrame(animate);}requestAnimationFrame(animate);
window.addEventListener('resize',()=>{drawLC();drawDust();});drawLC();drawDust();
