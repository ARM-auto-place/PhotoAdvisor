const {test} = require('node:test');
const assert = require('node:assert/strict');
const {describe, nextSteps} = require('../app/pilot_feedback.js');
const photo = {id:'one', filename:'01.jpg', decision:null};
const result = {candidate:'front', recognition:'uncertain', accepted:false, quality:'good', quality_issues:[]};

test('uncertain usable photo asks for confirmation without demanding a new photo', () => {
  const view = describe(photo, result, [photo]);
  assert.equal(view.canConfirm, true);
  assert.match(view.reasons.at(-1)[0], /подтверждение/);
  assert.notEqual(view.tone, 'good');
});
test('saved manual choice on a poor photo is never shown as counted', () => {
  const view = describe({...photo, decision:'front'}, {...result, confirmed:true, quality:'poor', quality_issues:['blurred']}, [photo]);
  assert.equal(view.title, 'Не засчитано');
  assert.equal(view.canConfirm, false);
  assert.match(view.reasons[0][1], /фокусировки/);
  assert.match(view.reasons.at(-1)[0], /сохранён/);
});
test('duplicate names the current representative, including after promotion', () => {
  const view = describe(photo, {...result, duplicate_of:'two'}, [photo, {id:'two', filename:'02.jpg'}]);
  assert.equal(view.title, 'Не засчитано');
  assert.equal(view.canConfirm, false);
  assert.match(view.reasons[0][1], /02.jpg/);
});
test('other is a saved exclusion, not a required view', () => {
  const view = describe({...photo, decision:'other'}, {...result, candidate:'other', confirmed:true}, [photo]);
  assert.equal(view.title, 'Не закрывает нужный ракурс');
  assert.equal(view.canConfirm, false);
  assert.notEqual(view.tone, 'good');
});
test('unknown and technical failures give different next actions', () => {
  const unknown = describe(photo, {...result, candidate:'other', recognition:'unknown'}, [photo]);
  assert.match(unknown.reasons[0][1], /вручную/);
  const failed = describe(photo, {...result, error:'inference_failed'}, [photo]);
  assert.equal(failed.canConfirm, false);
  assert.match(failed.reasons[0][1], /ещё раз/);
});
test('accepted photo and invalidated analysis are not confused', () => {
  assert.equal(describe({...photo, decision:'front'}, {...result, accepted:true}, [photo]).title, 'Ракурс засчитан');
  const pending = describe(photo, undefined, [photo]);
  assert.equal(pending.canConfirm, false);
  assert.match(pending.reasons[0][1], /актуальный результат/);
});
test('unknown quality issue has a safe explanation and repeated issues collapse', () => {
  const view = describe(photo, {...result, quality:'uncertain', quality_issues:['future_code','future_code']}, [photo]);
  assert.equal(view.canConfirm, false);
  assert.equal(view.reasons.length, 1);
  assert.match(view.reasons[0][1], /резкость/);
});
test('next steps prioritize existing photos, exclude covered and preserve report order', () => {
  const categories = [
    {id:'rear',priority:'required',status:'missing'},
    {id:'front',priority:'required',status:'uncertain'},
    {id:'interior',priority:'required',status:'present'},
    {id:'damage',priority:'recommended',status:'missing'},
  ];
  const before = JSON.stringify(categories);
  assert.deepEqual(nextSteps({categories}).map(c=>c.id), ['front','rear']);
  assert.equal(JSON.stringify(categories), before);
  assert.deepEqual(nextSteps({categories:[]}), []);
});
