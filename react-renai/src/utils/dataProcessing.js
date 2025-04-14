/**
 * Преобразует сырые данные эмбеддингов (предположительно массив массивов векторов)
 * в формат, ожидаемый компонентами (массив векторов).
 * @param {Array<Array<Array<number>>>} rawEmbeddings - Сырые данные, например [ [[vec1]], [[vec2]], ... ]
 * @returns {Array<Array<number>>} Обработанные эмбеддинги, например [ [vec1], [vec2], ... ]
 */
export const processRawEmbeddings = (rawEmbeddings) => {
    if (!Array.isArray(rawEmbeddings)) {
      console.error("processRawEmbeddings: ожидался массив, получено:", rawEmbeddings);
      return [];
    }
  
    return rawEmbeddings.map(item => {
      // Проверяем, что элемент является массивом и содержит хотя бы один вложенный массив (вектор)
      if (Array.isArray(item) && item.length > 0 && Array.isArray(item[0])) {
        return item[0]; // Извлекаем первый элемент (вектор)
      } else {
        console.warn("processRawEmbeddings: Некорректный формат элемента:", item);
        // Возвращаем пустой массив или null, чтобы обозначить проблему,
        // или можно отфильтровать такие элементы позже.
        return null; 
      }
    }).filter(vector => vector !== null); // Удаляем элементы, которые не удалось обработать
  };
 