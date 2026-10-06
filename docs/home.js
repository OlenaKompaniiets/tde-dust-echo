'use strict';
const stageTabs=[...document.querySelectorAll('[data-stage]')];
function selectStage(i,focus=false){stageTabs.forEach((tab,j)=>{tab.setAttribute('aria-selected',String(i===j));tab.tabIndex=i===j?0:-1;document.getElementById('stage-'+j).hidden=i!==j;});if(focus)stageTabs[i].focus();}
stageTabs.forEach((tab,i)=>{tab.addEventListener('click',()=>selectStage(i));tab.addEventListener('keydown',e=>{let j=i;if(e.key==='ArrowRight')j=(i+1)%stageTabs.length;else if(e.key==='ArrowLeft')j=(i+stageTabs.length-1)%stageTabs.length;else if(e.key==='Home')j=0;else if(e.key==='End')j=stageTabs.length-1;else return;e.preventDefault();selectStage(j,true);});});
