import React, { useEffect, useState, useMemo } from 'react';
import { 
  Box, Paper, Typography, Grid, FormControl,
  InputLabel, Select, MenuItem, CircularProgress,
  Tooltip, Chip
} from '@mui/material';
import { HeatMap } from '@nivo/heatmap';

const SimilarityHeatmap = ({ embeddings, artworkMetadata }) => {
  const [loading, setLoading] = useState(true);
  const [groupBy, setGroupBy] = useState('style');
  const [sortBy, setSortBy] = useState('similarity');
  const [heatmapData, setHeatmapData] = useState([]);
  
  // Расчет матрицы сходства
  useEffect(() => {
    if (!embeddings || !artworkMetadata) return;
    
    setLoading(true);
    
    const calculateSimilarityMatrix = () => {
      // Сгруппировать произведения по выбранному атрибуту
      const groupedWorks = {};
      
      artworkMetadata.forEach((meta, idx) => {
        const groupValue = meta[groupBy] || 'Unknown';
        
        if (!groupedWorks[groupValue]) {
          groupedWorks[groupValue] = [];
        }
        
        groupedWorks[groupValue].push({
          id: idx,
          metadata: meta,
          embedding: embeddings[idx]
        });
      });
      
      // Рассчитать среднее сходство между группами
      const groups = Object.keys(groupedWorks);
      const matrix = [];
      
      groups.forEach(group1 => {
        const row = {
          id: group1,
          data: []
        };
        
        groups.forEach(group2 => {
          let totalSimilarity = 0;
          let comparisons = 0;
          
          // Сравнить все работы из group1 со всеми работами из group2
          groupedWorks[group1].forEach(work1 => {
            groupedWorks[group2].forEach(work2 => {
              if (work1.id !== work2.id) {
                // Рассчитать косинусное сходство
                let dotProduct = 0;
                let norm1 = 0;
                let norm2 = 0;
                
                for (let i = 0; i < work1.embedding.length; i++) {
                  dotProduct += work1.embedding[i] * work2.embedding[i];
                  norm1 += work1.embedding[i] * work1.embedding[i];
                  norm2 += work2.embedding[i] * work2.embedding[i];
                }
                
                norm1 = Math.sqrt(norm1);
                norm2 = Math.sqrt(norm2);
                
                const similarity = dotProduct / (norm1 * norm2);
                totalSimilarity += similarity;
                comparisons++;
              }
            });
          });
          
          const avgSimilarity = comparisons > 0 ? totalSimilarity / comparisons : 0;
          
          row.data.push({
            x: group2,
            y: avgSimilarity
          });
        });
        
        matrix.push(row);
      });
      
      // Сортировка по сходству если выбрано
      if (sortBy === 'similarity') {
        // Рассчитать средние значения сходства для каждой группы
        const groupAverageSimilarity = {};
        
        matrix.forEach(row => {
          let sum = 0;
          row.data.forEach(cell => {
            sum += cell.y;
          });
          groupAverageSimilarity[row.id] = sum / row.data.length;
        });
        
        // Сортировка групп по убыванию среднего сходства
        matrix.sort((a, b) => groupAverageSimilarity[b.id] - groupAverageSimilarity[a.id]);
        
        // Переупорядочить данные по той же сортировке
        const orderedGroups = matrix.map(row => row.id);
        matrix.forEach(row => {
          row.data.sort((a, b) => {
            return orderedGroups.indexOf(a.x) - orderedGroups.indexOf(b.x);
          });
        });
      }
      
      return matrix;
    };
    
    // Запустить расчет в неблокирующем режиме
    setTimeout(() => {
      try {
        const matrix = calculateSimilarityMatrix();
        setHeatmapData(matrix);
      } finally {
        setLoading(false);
      }
    }, 0);
  }, [embeddings, artworkMetadata, groupBy, sortBy]);
  
  // Цветовая схема
  const colors = useMemo(() => {
    return {
      scheme: 'blues',
      minValue: 0.5,
      maxValue: 1
    };
  }, []);
  
  return (
    <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h5" gutterBottom>
        Тепловая карта сходства
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Группировать по</InputLabel>
            <Select
              value={groupBy}
              onChange={(e) => setGroupBy(e.target.value)}
              label="Группировать по"
            >
              <MenuItem value="era">Эпоха</MenuItem>
              <MenuItem value="style">Стиль</MenuItem>
              <MenuItem value="region">Регион</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Сортировка</InputLabel>
            <Select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              label="Сортировка"
            >
              <MenuItem value="similarity">По сходству</MenuItem>
              <MenuItem value="alphabetical">По алфавиту</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
          <CircularProgress />
        </Box>
      ) : heatmapData.length > 0 ? (
        <Box sx={{ flex: 1, minHeight: 600 }}>
          <HeatMap
            data={heatmapData}
            margin={{ top: 40, right: 90, bottom: 80, left: 90 }}
            valueFormat=".2f"
            axisTop={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: -45,
              legend: '',
              legendOffset: 46
            }}
            axisRight={null}
            axisBottom={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: -45,
              legend: '',
              legendPosition: 'middle',
              legendOffset: 46
            }}
            axisLeft={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              legend: '',
              legendPosition: 'middle',
              legendOffset: -72
            }}
            colors={{
              type: 'sequential',
              scheme: 'blues',
              minValue: 0.5,
              maxValue: 1
            }}
            emptyColor="#eeeeee"
            labelTextColor={{ from: 'color', modifiers: [['darker', 2]] }}
            legends={[
              {
                anchor: 'bottom',
                translateX: 0,
                translateY: 60,
                length: 400,
                thickness: 8,
                direction: 'row',
                tickPosition: 'after',
                tickSize: 3,
                tickSpacing: 4,
                tickOverlap: false,
                tickFormat: '.2f',
                title: 'Сходство →',
                titleAlign: 'start',
                titleOffset: 4
              }
            ]}
            annotations={[
              {
                type: 'rect',
                match: { id: 'Ancient Greek', value: 'Ancient Roman' },
                noteTextOffset: 4,
                offset: 3,
                noteWidth: 120,
                noteHeight: 60,
                note: 'Высокое сходство между искусством Древней Греции и Рима',
                noteAlign: 'middle',
                noteBgColor: 'rgba(255, 255, 255, 0.9)',
                noteTextColor: '#333333'
              }
            ]}
            hoverTarget="cell"
            tooltip={({ xKey, yKey, value, color }) => (
              <Box
                sx={{
                  bgcolor: 'background.paper',
                  p: 1.5,
                  borderRadius: 1,
                  boxShadow: 3,
                  maxWidth: 300
                }}
              >
                <Typography variant="subtitle2">
                  Сравнение: {xKey} / {yKey}
                </Typography>
                <Typography variant="body2">
                  Сходство: {value.toFixed(4)}
                </Typography>
              </Box>
            )}
          />
        </Box>
      ) : (
        <Typography variant="body1">Нет данных для отображения</Typography>
      )}
    </Paper>
  );
};

export default SimilarityHeatmap;
