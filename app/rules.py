def summarize(listing_id, photos, checklist, locale, model_version, calibrated):
    # Three-quarter views may map to at most one category, as configured explicitly.
    mapping = checklist['coverage']
    categories = []
    suggestions = []
    texts = checklist['messages'].get(locale, checklist['messages']['ru'])
    for category in checklist['categories']:
        cid = category['id']
        matches = [p for p in photos if p.accepted and mapping.get(p.category) == cid]
        unclear = [p for p in photos if not p.accepted and p.recognition in {'known', 'uncertain'}
                   and not p.duplicate_of and mapping.get(p.candidate) == cid]
        state = 'present' if matches else ('uncertain' if unclear else 'missing')
        title = category['title'].get(locale, category['title']['ru'])
        hint = category['hint'].get(locale, category['hint']['ru'])
        categories.append(dict(id=cid, priority=category['priority'], title=title, hint=hint,
                               status=state, photo_ids=[p.photo_id for p in matches],
                               uncertain_photo_ids=[p.photo_id for p in unclear]))
        if state != 'present':
            suggestions.append(dict(category=cid, priority=category['priority'],
                                    message=hint, conditional=category.get('conditional', False)))
    groups = {}
    for priority in ('required', 'recommended'):
        groups[priority] = {state: [c['id'] for c in categories
                                   if c['priority'] == priority and c['status'] == state]
                            for state in ('present', 'missing', 'uncertain')}
    errors = sum(p.error is not None for p in photos)
    if photos and errors == len(photos):
        status = 'failed'
    elif groups['required']['missing']:
        status = 'incomplete'
    elif groups['required']['uncertain']:
        status = 'needs_review'
    else:
        status = 'complete'
    # Technical problems and unresolved recognitions take precedence over apparent completeness.
    if status != 'failed' and (errors or any(
            not p.duplicate_of and (p.recognition in {'unknown', 'uncertain'}
                                   or p.quality in {'poor', 'uncertain'}) for p in photos)):
        status = 'needs_review'
    notices = []
    if any(p.duplicate_of for p in photos):
        notices.append(texts['duplicates'])
    if any(p.quality == 'poor' for p in photos):
        notices.append(texts['quality'])
    if any(not p.duplicate_of and p.recognition in {'unknown', 'uncertain'} for p in photos):
        notices.append(texts['uncertain'])
    return dict(listing_id=listing_id, status=status,
                processing_status='failed' if status == 'failed' else ('partial' if errors else 'succeeded'),
                **groups, categories=categories, photo_results=[p.model_dump() for p in photos],
                suggestions=suggestions, notices=notices,
                model={'version': model_version, 'calibrated': calibrated})
