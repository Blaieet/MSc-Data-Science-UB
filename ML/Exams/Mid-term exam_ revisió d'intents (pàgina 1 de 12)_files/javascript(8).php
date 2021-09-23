
/**
 * This plugin provides access to Moodle data in form of analytics and reports in real time.
 *
 *
 * @package    local_intelliboard
 * @copyright  2017 IntelliBoard, Inc
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 * @website    https://intelliboard.net/
 */
var intelliboardAjax=30;var intelliboardAjaxUrl="";var intelliboardAjaxCounter=0;var intelliboardCounter=0;var intelliboardInactivity=60;var intelliboardPeriod=1000;var intelliboardInterval=null;var intelliboardPage='';var intelliboardParam='';var intelliboardTime=0;var intelliboardMediaTrack=0;var intelliboardSSOLink=!1;function intelliboardInit(Y,options){options=options||{};intelliboardAjax=options.intelliboardAjax||intelliboardAjax;intelliboardAjaxUrl=options.intelliboardAjaxUrl||intelliboardAjaxUrl;intelliboardInactivity=options.intelliboardInactivity||intelliboardInactivity;intelliboardPeriod=options.intelliboardPeriod||intelliboardPeriod;intelliboardInterval=setInterval(intelliboardProgress,intelliboardPeriod);intelliboardPage=options.intelliboardPage||intelliboardPage;intelliboardParam=options.intelliboardParam||intelliboardParam;intelliboardTime=options.intelliboardTime||intelliboardTime;intelliboardMediaTrack=options.intelliboardMediaTrack||intelliboardTime;intelliboardSSOLink=options.intelliboardSSOLink||intelliboardSSOLink;if(intelliboardSSOLink){try{var links=document.querySelectorAll("a[href='"+intelliboardSSOLink+"']");if(links.length>0){links.forEach(function(link){link.setAttribute("target","_blank")})}}catch(error){}}}
function intelliboardProgress(){if(intelliboardMediaTrack){var status=intelliboardMedia();if(status&&!document.hidden){clearIntelliboardCounter()}}
if(!document.hidden&&intelliboardCounter<=intelliboardInactivity){intelliboardTime++;intelliboardCounter++;intelliboardAjaxCounter++;if(intelliboardAjaxCounter==intelliboardAjax&&intelliboardAjaxUrl&&intelliboardAjax){sendIntelliboardTime(intelliboardTime);intelliboardAjaxCounter=0}}}
function intelliboardMedia(){var media=[];var status=!1;var internal=document.querySelectorAll('audio,video');var frames=document.querySelectorAll('iframe');if(frames.length){frames.forEach(function(frame){var elements=frame.contentWindow.document.querySelectorAll('audio,video');if(elements.length){elements.forEach(function(element){media.push(element)})}})}
if(internal.length){internal.forEach(function(element){media.push(element)})}
if(media.length){media.forEach(function(element){if(!element.paused){status=!0}})}
return status}
if(document.addEventListener){document.addEventListener("mousemove",clearIntelliboardCounter);document.addEventListener("keypress",clearIntelliboardCounter);document.addEventListener("scroll",clearIntelliboardCounter);window.addEventListener("beforeunload",resetIntelliboardParams)}else if(document.attachEvent){document.attachEvent("mousemove",clearIntelliboardCounter);document.attachEvent("keypress",clearIntelliboardCounter);document.attachEvent("scroll",clearIntelliboardCounter);window.addEventListener("beforeunload",resetIntelliboardParams)}
function sendIntelliboardTime(time){if(!time){return}
var xmlhttp;if(window.XMLHttpRequest){xmlhttp=new XMLHttpRequest()}else{xmlhttp=new ActiveXObject("Microsoft.XMLHTTP")}
xmlhttp.onreadystatechange=function(){if(xmlhttp.readyState==4){if(xmlhttp.status==200){intelliboardTime=intelliboardTime-time}}}
resetIntelliboardParams();xmlhttp.open("GET",intelliboardAjaxUrl,!0);xmlhttp.send()}
function resetIntelliboardParams(){setIntelliboardCookie('intelliboardPage',intelliboardPage);setIntelliboardCookie('intelliboardParam',intelliboardParam);setIntelliboardCookie('intelliboardTime',intelliboardTime)}
function clearIntelliboardCounter(){intelliboardCounter=0;intelliboardWarningTime=0;intelliboardLogoutTime=0}
function getIntelliboardCookie(name){var matches=document.cookie.match(new RegExp("(?:^|; )"+name.replace(/([\.$?*|{}\(\)\[\]\\\/\+^])/g,'\\$1')+"=([^;]*)"));return matches?decodeURIComponent(matches[1]):0}
function setIntelliboardCookie(name,value,options){options=options||{};var expires=options.expires;if(typeof expires=="number"&&expires){var d=new Date();d.setTime(d.getTime()+expires*1000);expires=options.expires=d}
if(expires&&expires.toUTCString){options.expires=expires.toUTCString()}
options.path="/";value=encodeURIComponent(value);var updatedCookie=name+"="+value+(location.protocol==='https:'?';secure':'');for(var propName in options){updatedCookie+="; "+propName;var propValue=options[propName];if(propValue!==!0){updatedCookie+="="+propValue}}
document.cookie=updatedCookie}
function deleteIntelliboardCookie(name){setIntelliboardCookie(name,"",{expires:-1})}