import React, { useState, useEffect } from 'react';
import { 
  Box, Paper, Typography, Grid, Tabs, Tab, 
  CircularProgress, Chip, Slider, CardMedia, Card
} from '@mui/material';
import { Radar } from 'react-chartjs-2';
import { Doughnut } from 'react-chartjs-2';
import { PolarArea } from 'react-chartjs-2';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend, ArcElement } from 'chart.js';

// Регистрация компонентов Chart.js
ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend, ArcElement);

const StyleAnalysis = ({ artworks, embeddings }) => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedArtwork, setSelectedArtwork] = useState(0);
  
  // Мокап данных для визуализации стилистических особенностей
  const styleFeaturesData = {
    labels: [
      'Геометричность',
      'Натурализм',
      'Экспрессивность',
      'Абстрактность',
      'Символизм',
      'Декоративность'
    ],
    datasets: [
      {
        label: 'Стилистический профиль',
        data: [0.8, 0.5, 0.3, 0.2, 0.9, 0.7],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(255, 99, 132, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(255, 99, 132, 1)',
        pointLabelFontSize: 14
      }
    ]
  };
  
  // Мокап данных для цветового анализа
  const colorAnalysisData = {
    labels: ['Красный', 'Охра', 'Коричневый', 'Черный', 'Белый'],
    datasets: [
      {
        data: [30, 25, 20, 15, 10],
        backgroundColor: [
          'rgba(255, 99, 132, 0.7)',
          'rgba(255, 159, 64, 0.7)',
          'rgba(165, 42, 42, 0.7)',
          'rgba(54, 57, 64, 0.7)',
          'rgba(250, 250, 250, 0.7)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(255, 159, 64, 1)',
          'rgba(165, 42, 42, 1)',
          'rgba(54, 57, 64, 1)',
          'rgba(250, 250, 250, 1)'
        ],
        borderWidth: 1
      }
    ]
  };
  
  // Мокап данных для анализа композиции
  const compositionAnalysisData = {
    labels: [
      'Симметрия',
      'Горизонтальные линии',
      'Вертикальные линии',
      'Диагональные элементы',
      'Фронтальная проекция',
      'Ритмичность'
    ],
    datasets: [
      {
        data: [0.9, 0.7, 0.8, 0.3, 0.9, 0.6],
        backgroundColor: [
          'rgba(54, 162, 235, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
          'rgba(255, 205, 86, 0.5)',
          'rgba(255, 99, 132, 0.5)',
          'rgba(201, 203, 207, 0.5)'
        ],
        borderWidth: 0
      }
    ]
  };
  
  const handleChangeTab = (event, newValue) => {
    setSelectedTab(newValue);
  };
  
  const handleSelectArtwork = (index) => {
    setSelectedArtwork(index);
  };
  
  if (!artworks || artworks.length === 0) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6">Нет доступных произведений для анализа</Typography>
      </Paper>
    );
  }
  
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Анализ художественных особенностей
      </Typography>
      
      <Tabs
        value={selectedTab}
        onChange={handleChangeTab}
        variant="fullWidth"
        indicatorColor="primary"
        textColor="primary"
        sx={{ mb: 3 }}
      >
        <Tab label="Стилистический анализ" />
        <Tab label="Цветовой анализ" />
        <Tab label="Композиционный анализ" />
      </Tabs>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Typography variant="h6" gutterBottom>
            Выберите произведение
          </Typography>
          
          <Box sx={{ maxHeight: 600, overflowY: 'auto' }}>
            {artworks.map((artwork, index) => (
              <Card 
                key={index}
                sx={{ 
                  mb: 2, 
                  cursor: 'pointer',
                  border: selectedArtwork === index ? '2px solid #3f51b5' : 'none',
                  transition: 'all 0.2s'
                }}
                onClick={() => handleSelectArtwork(index)}
              >
                <CardMedia
                  component="img"
                  height="140"
                  image={artwork.imageUrl}
                  alt={artwork.title}
                />
                <Box sx={{ p: 1.5 }}>
                  <Typography variant="subtitle1" noWrap>
                    {artwork.title || `Произведение ${index + 1}`}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 1 }}>
                    {artwork.era && (
                      <Chip 
                        label={artwork.era} 
                        size="small" 
                        color="primary" 
                        variant="outlined"
                      />
                    )}
                    {artwork.style && (
                      <Chip 
                        label={artwork.style} 
                        size="small"
                        color="secondary"
                        variant="outlined"
                      />
                    )}
                  </Box>
                </Box>
              </Card>
            ))}
          </Box>
        </Grid>
        
        <Grid item xs={12} md={8}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              {selectedTab === 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Стилистический профиль
                  </Typography>
                  <Typography variant="body2" paragraph>
                    Радарная диаграмма отображает выраженность различных стилистических характеристик в произведении.
                    Данный профиль показывает, что в выбранном произведении наиболее выражены геометричность и символизм,
                    что характерно для древнеегипетского искусства.
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Radar 
                      data={styleFeaturesData}
                      options={{
                        scales: {
                          r: {
                            beginAtZero: true,
                            min: 0,
                            max: 1,
                            ticks: {
                              stepSize: 0.2
                            }
                          }
                        },
                        plugins: {
                          legend: {
                            position: 'bottom'
                          }
                        },
                        elements: {
                          line: {
                            tension: 0.2
                          }
                        }
                      }}
                    />
                  </Box>
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Модель идентифицирует в данном произведении ключевые черты канонического египетского стиля:
                    фронтальное изображение, символическое представление фигур, иерархическое масштабирование 
                    и плоскостное изображение без передачи глубины пространства.
                  </Typography>
                </Box>
              )}
              
              {selectedTab === 1 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Цветовой анализ
                  </Typography>
                  <Typography variant="body2" paragraph>
                    Диаграмма показывает распределение основных цветов в произведении. Древнеегипетская настенная 
                    живопись характеризуется использованием ограниченной палитры из красных, охристых и коричневых оттенков,
                    что отчетливо распознается моделью.
                  </Typography>
                  <Box sx={{ height: 400, display: 'flex', justifyContent: 'center' }}>
                    <Box sx={{ width: 400 }}>
                      <Doughnut 
                        data={colorAnalysisData}
                        options={{
                          plugins: {
                            legend: {
                              position: 'bottom'
                            }
                          },
                          cutout: '50%'
                        }}
                      />
                    </Box>
                  </Box>
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Цветовой анализ модели согласуется с историческими данными о красках, использовавшихся 
                    в египетском искусстве. Красные и коричневые пигменты получали из охры и гематита,
                    что обусловило характерную цветовую гамму древнеегипетской живописи.
                  </Typography>
                </Box>
              )}
              
              {selectedTab === 2 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Композиционный анализ
                  </Typography>
                  <Typography variant="body2" paragraph>
                    Полярная диаграмма отображает основные композиционные характеристики произведения.
                    Анализ показывает высокую степень симметрии, преобладание вертикальных линий
                    и использование фронтальных проекций, типичных для египетского изобразительного канона.
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <PolarArea 
                      data={compositionAnalysisData}
                      options={{
                        scales: {
                          r: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                              display: false
                            }
                          }
                        },
                        plugins: {
                          legend: {
                            position: 'right'
                          }
                        }
                      }}
                    />
                  </Box>
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Модель обнаруживает особый подход к построению композиции в древнеегипетском искусстве:
                    сочетание профильного изображения ног и головы с фронтальным изображением торса.
                    Такая комбинация проекций является узнаваемой характеристикой, которую ИИ успешно идентифицирует.
                  </Typography>
                </Box>
              )}
            </>
          )}
        </Grid>
      </Grid>
    </Paper>
  );
};

export default StyleAnalysis;
