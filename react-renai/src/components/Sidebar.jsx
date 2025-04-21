import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Drawer, List, ListItem, ListItemIcon, ListItemText, 
  Collapse, Divider, Typography, Box, ListSubheader,
  ListItemButton, Button, CircularProgress
} from '@mui/material';
import {
  Dashboard, BubbleChart, Visibility, Code, 
  AccountTree, Settings, Compare, History, 
  ExpandLess, ExpandMore, Palette, Category,
  ArtTrack,
  ScatterPlot
} from '@mui/icons-material';

const Sidebar = ({ artworks = [], onArtworkSelect, isLoadingMore, hasMoreArtworks, onLoadMore }) => {
  const navigate = useNavigate();
  const [open, setOpen] = useState({});
  const [openArtworks, setOpenArtworks] = useState(true);
  
  const toggleSubmenu = (key) => {
    setOpen(prev => ({ ...prev, [key]: !prev[key] }));
  };
  
  const handleNavigate = (path) => {
    navigate(path);
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
        <ListItemButton onClick={() => handleNavigate('/overview')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Dashboard />
          </ListItemIcon>
          <ListItemText primary="Overview" />
        </ListItemButton>
        
        <ListItemButton onClick={() => handleNavigate('/embeddings')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <ScatterPlot />
          </ListItemIcon>
          <ListItemText primary="Embedding Projection" />
        </ListItemButton>
        
        {/* Visual Attention */}
        <ListItemButton onClick={() => toggleSubmenu('attention')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Visibility />
          </ListItemIcon>
          <ListItemText primary="Visual Attention" />
          {open.attention ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
        
        <Collapse in={open.attention || false} timeout="auto">
          <List component="div" disablePadding>
            <ListItemButton sx={{ pl: 4 }} onClick={() => handleNavigate('/attention')}>
              <ListItemText primary="Attention Maps" />
            </ListItemButton>
          </List>
        </Collapse>
        
        {/* Art Similarities */}
        <ListItemButton onClick={() => toggleSubmenu('similarities')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Compare />
          </ListItemIcon>
          <ListItemText primary="Artwork Similarities" />
          {open.similarities ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
        
        <Collapse in={open.similarities || false} timeout="auto">
          <List component="div" disablePadding>
            <ListItemButton sx={{ pl: 4 }} onClick={() => handleNavigate('/heatmap')}>
              <ListItemText primary="Heatmap" />
            </ListItemButton>
            <ListItemButton sx={{ pl: 4 }} onClick={() => handleNavigate('/similarity')}>
              <ListItemText primary="Graph of Connections" />
            </ListItemButton>
          </List>
        </Collapse>
        
        {/* Artistic Features */}
        <ListItemButton onClick={() => toggleSubmenu('features')}>
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Palette />
          </ListItemIcon>
          <ListItemText primary="Artistic Features" />
          {open.features ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
        
        <Collapse in={open.features || false} timeout="auto">
          <List component="div" disablePadding>
            <ListItemButton sx={{ pl: 4 }} onClick={() => handleNavigate('/analysis')}>
              <ListItemText primary="Style Analysis" />
            </ListItemButton>
          </List>
        </Collapse>
        
        {/* Settings */}
        {/* <ListItemButton onClick={() => handleNavigate('/settings')}> */}
        {/*   <ListItemIcon sx={{ color: '#6c7293' }}><Settings /></ListItemIcon> */}
        {/*   <ListItemText primary="Настройки" /> */}
        {/* </ListItemButton> */}

        {/* Artworks List */}
        <Divider sx={{ bgcolor: 'rgba(255,255,255,0.1)', my: 1 }} />
        <ListItem 
          onClick={() => setOpenArtworks(!openArtworks)} 
          sx={{ cursor: 'pointer' }} 
        >
          <ListItemIcon sx={{ color: '#6c7293' }}>
            <Category /> 
          </ListItemIcon>
          <ListItemText primary="Artworks" />
          {openArtworks ? <ExpandLess /> : <ExpandMore />}
        </ListItem>
        <Collapse in={openArtworks} timeout="auto">
          <Box sx={{ maxHeight: 400, overflowY: 'auto', pr: 1 }}>
            <List component="div" disablePadding>
              {artworks.map((artwork, index) => (
                <ListItemButton 
                  key={`sidebar-artwork-${artwork.id || index}`}
                  sx={{ pl: 4 }} 
                  onClick={() => onArtworkSelect && onArtworkSelect(artwork.id)}
                >
                  <ListItemIcon sx={{ color: '#adb5bd', minWidth: '30px' }}>
                    <ArtTrack fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary={artwork.title || `Object ${artwork.id}`}
                    primaryTypographyProps={{ 
                        style: { fontSize: '0.9rem', color: '#adb5bd', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } 
                    }} 
                  />
                </ListItemButton>
              ))}
            </List>
            {hasMoreArtworks && (
              <Box sx={{ textAlign: 'center', py: 1 }}>
                <Button 
                  size="small" 
                  onClick={onLoadMore} 
                  disabled={isLoadingMore}
                  variant="outlined"
                  sx={{ color: '#adb5bd', borderColor: '#adb5bd' }}
                >
                  {isLoadingMore ? <CircularProgress size={20} color="inherit" /> : 'Загрузить еще'}
                </Button>
              </Box>
            )}
          </Box>
        </Collapse>

      </List>
    </Drawer>
  );
};

export default Sidebar;
