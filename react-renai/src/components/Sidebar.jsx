import React, { useState } from 'react';
import { 
  Drawer, List, ListItem, ListItemIcon, ListItemText, 
  Collapse, Divider, Typography, Box
} from '@mui/material';
import {
  Dashboard, BubbleChart, Visibility, Code, 
  AccountTree, Settings, Compare, History, 
  ExpandLess, ExpandMore, Palette, Category
} from '@mui/icons-material';

const Sidebar = ({ onNavigate }) => {
  const [open, setOpen] = useState({});
  
  const toggleSubmenu = (key) => {
    setOpen(prev => ({ ...prev, [key]: !prev[key] }));
  };
  
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: 280,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 280,
          background: 'linear-gradient(180deg, #1a2035 0%, #232b45 100%)',
          color: '#fff',
          borderRight: 'none'
        }
      }}
    >
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h5" sx={{ 
          fontWeight: 300,
          background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          AI Art Vision
        </Typography>
      </Box>
      
      <Divider sx={{ bgcolor: 'rgba(255,255,255,0.1)' }} />
      
      <List>
        <ListItem button onClick={() => onNavigate('dashboard')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Dashboard />
          </ListItemIcon>
          <ListItemText primary="Обзор" />
        </ListItem>
        
        {/* Embedding Analysis */}
        <ListItem button onClick={() => toggleSubmenu('embeddings')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Code />
          </ListItemIcon>
          <ListItemText primary="Анализ эмбеддингов" />
          {open.embeddings ? <ExpandLess /> : <ExpandMore />}
        </ListItem>
        
        <Collapse in={open.embeddings || false} timeout="auto">
          <List component="div" disablePadding>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('embeddingProjection')}>
              <ListItemText primary="t-SNE / UMAP проекция" />
            </ListItem>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('embeddingClusters')}>
              <ListItemText primary="Кластеризация" />
            </ListItem>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('embeddingNearest')}>
              <ListItemText primary="Поиск похожих" />
            </ListItem>
          </List>
        </Collapse>
        
        {/* Visual Attention */}
        <ListItem button onClick={() => toggleSubmenu('attention')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Visibility />
          </ListItemIcon>
          <ListItemText primary="Визуальное внимание" />
          {open.attention ? <ExpandLess /> : <ExpandMore />}
        </ListItem>
        
        <Collapse in={open.attention || false} timeout="auto">
          <List component="div" disablePadding>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('attentionMaps')}>
              <ListItemText primary="Карты внимания" />
            </ListItem>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('gradcamVisualization')}>
              <ListItemText primary="GradCAM" />
            </ListItem>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('attentionComparison')}>
              <ListItemText primary="Сравнение" />
            </ListItem>
          </List>
        </Collapse>
        
        {/* Art Similarities */}
        <ListItem button onClick={() => toggleSubmenu('similarities')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Compare />
          </ListItemIcon>
          <ListItemText primary="Сходство произведений" />
          {open.similarities ? <ExpandLess /> : <ExpandMore />}
        </ListItem>
        
        <Collapse in={open.similarities || false} timeout="auto">
          <List component="div" disablePadding>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('heatmap')}>
              <ListItemText primary="Тепловая карта" />
            </ListItem>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('networkGraph')}>
              <ListItemText primary="Граф связей" />
            </ListItem>
          </List>
        </Collapse>
        
        {/* Artistic Features */}
        <ListItem button onClick={() => toggleSubmenu('features')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Palette />
          </ListItemIcon>
          <ListItemText primary="Художественные особенности" />
          {open.features ? <ExpandLess /> : <ExpandMore />}
        </ListItem>
        
        <Collapse in={open.features || false} timeout="auto">
          <List component="div" disablePadding>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('styleAnalysis')}>
              <ListItemText primary="Анализ стилей" />
            </ListItem>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('compositionAnalysis')}>
              <ListItemText primary="Композиция" />
            </ListItem>
            <ListItem button sx={{ pl: 4 }} onClick={() => onNavigate('colorAnalysis')}>
              <ListItemText primary="Цветовой анализ" />
            </ListItem>
          </List>
        </Collapse>
        
        {/* Settings */}
        <ListItem button onClick={() => onNavigate('settings')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Settings />
          </ListItemIcon>
          <ListItemText primary="Настройки" />
        </ListItem>
      </List>
    </Drawer>
  );
};

export default Sidebar;
