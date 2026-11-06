/* Presentation only: acceptance and duplicate decisions come from Advisor. */
const pilotFeedback = (() => {
  const quality = {
    too_small: ['Мало деталей', 'Загрузите исходное фото покрупнее или переснимите. Не увеличивайте маленькую копию.'],
    blurred: ['Фото нечёткое', 'Протрите объектив, дождитесь фокусировки и снимите ещё раз, удерживая телефон неподвижно.'],
    underexposed: ['Слишком темно', 'Переснимите при дневном свете или добавьте равномерное освещение.'],
    overexposed: ['Слишком светло', 'Уберите блики и переснимите без прямого солнца в объектив.'],
    multiple_cars: ['В кадре несколько машин', 'Снимите только нужный автомобиль, чтобы другие машины не попадали в кадр.'],
    car_not_detected: ['Не удалось выделить машину', 'Проверьте, что выбран нужный снимок. Для общего вида снимите машину целиком при хорошем свете.'],
    car_too_small: ['Машина слишком далеко', 'Подойдите ближе: автомобиль должен занимать большую часть кадра.'],
    possible_crop: ['Возможно, не всё попало в кадр', 'Для общего вида покажите машину целиком. Для крыла нужен крупный план арки с соседними деталями.'],
  };
  const errors = {
    invalid_image: 'Файл повреждён. Откройте оригинал и загрузите его заново.',
    unsupported_format: 'Загрузите обычное фото JPEG, PNG или WEBP.',
    unsupported_image: 'Загрузите обычное фото JPEG, PNG или WEBP без анимации.',
    too_many_pixels: 'Уменьшите разрешение исходного фото и загрузите его заново.',
    image_too_large: 'Загрузите фотографию меньшего размера.',
    inference_failed: 'Не удалось выполнить распознавание. Нажмите «Проверить ракурсы» ещё раз.',
    s3_unavailable: 'Фото временно недоступно. Повторите проверку.',
    s3_object_unavailable: 'Фото недоступно. Добавьте оригинал ещё раз и повторите проверку.',
  };

  function describe(photo, result, photos) {
    if (!result) return {
      title: 'Ожидает проверки', tone: 'warn', canConfirm: false,
      reasons: [['Набор изменился', 'Нажмите «Проверить ракурсы», чтобы получить актуальный результат.']],
    };
    const reasons = [];
    if (result.error) reasons.push(['Фото не обработано', errors[result.error]
      || 'Не удалось прочитать или обработать фото. Повторите проверку; если ошибка остаётся, загрузите оригинал заново.']);
    for (const issue of new Set(result.quality_issues || [])) {
      reasons.push(quality[issue] || ['Нужно проверить качество', 'Проверьте резкость, освещение и полноту кадра. При необходимости переснимите.']);
    }
    if (result.quality !== 'good' && !result.error && !reasons.length) {
      reasons.push(['Качество требует проверки', 'Проверьте резкость и освещение. Ручная метка не отменяет ограничения качества.']);
    }
    if (result.duplicate_of) {
      const original = photos.find(p => p.id === result.duplicate_of);
      reasons.push(['Повтор другого фото', 'В одной группе копий засчитывается только один снимок. '
        + (original ? 'Основной снимок группы: «' + original.filename + '».' : 'Проверьте основной снимок группы.')]);
    }
    const usable = !result.error && result.quality === 'good' && !result.duplicate_of;
    if (result.accepted) return {title: 'Ракурс засчитан', tone: 'good', reasons, canConfirm: false};
    if (!usable) {
      if (photo.decision) reasons.push(['Выбор сохранён', 'Этот снимок не закрывает ракурс. Сначала устраните указанную причину.']);
      return {title: 'Не засчитано', tone: 'warn', reasons, canConfirm: false};
    }
    if (photo.decision === 'other') return {
      title: 'Не закрывает нужный ракурс', tone: '', reasons: [
        ['Отмечено вами', 'Фото остаётся в проверке. Для недостающего ракурса добавьте другой снимок.'],
      ], canConfirm: false,
    };
    if (!result.candidate || result.candidate === 'other' || result.recognition === 'unknown') {
      reasons.push(['Ракурс не определён', 'Если фото подходит, выберите ракурс вручную. Иначе отметьте «Не подходит» или добавьте другой снимок.']);
    } else {
      reasons.push(['Нужно ваше подтверждение', 'Сравните фото с предложенным ракурсом. Подтвердите его или выберите правильный вариант ниже.']);
    }
    return {title: 'Проверьте ракурс', tone: 'warn', reasons, canConfirm: true};
  }

  function nextSteps(report) {
    return report.categories.filter(c => c.priority === 'required' && c.status !== 'present')
      .sort((a, b) => Number(b.status === 'uncertain') - Number(a.status === 'uncertain'));
  }
  return {describe, nextSteps};
})();
if (typeof module !== 'undefined') module.exports = pilotFeedback;
