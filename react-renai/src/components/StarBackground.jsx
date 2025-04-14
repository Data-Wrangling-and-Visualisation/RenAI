import React, { useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass";

const ThreeScene = () => {
  const mountRef = useRef();

  // Генерация цветов звезд с градиентом
  const starColors = useMemo(() => {
    const colors = [];
    const colorPalette = [
      new THREE.Color(0x8ab4f8), // Голубой
      new THREE.Color(0xffffff), // Белый
      new THREE.Color(0xffd700), // Золотой
    ];
    
    for (let i = 0; i < 5000; i++) {
      colors.push(colorPalette[Math.floor(Math.random() * colorPalette.length)]);
    }
    return colors;
  }, []);

  useEffect(() => {
    const container = mountRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Инициализация сцены
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000033);
    
    // Настройка камеры
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 50, 100);
    
    // Настройка рендерера
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Постобработка с эффектом свечения
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height),
      1.5, // Сила свечения
      0.4, // Радиус
      0.85 // Порог
    );
    composer.addPass(bloomPass);

    // Создание звездного поля с вариациями размеров
    const starsGeometry = new THREE.BufferGeometry();
    const positions = [];
    const sizes = [];
    
    for (let i = 0; i < 5000; i++) {
      positions.push(
        (Math.random() - 0.5) * 2000,
        (Math.random() - 0.5) * 2000,
        (Math.random() - 0.5) * 2000
      );
      sizes.push(Math.random() * 1.5 + 0.5);
    }
    
    starsGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3)
    );
    starsGeometry.setAttribute(
      "size",
      new THREE.Float32BufferAttribute(sizes, 1)
    );
    starsGeometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(
        starColors.flatMap(c => c.toArray()),
        3
      )
    );

    // Материал звезд с мерцанием
    const starsMaterial = new THREE.PointsMaterial({
      size: 0.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true
    });
    
    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);

    // Анимация мерцания
    const clock = new THREE.Clock();
    
    const animateStars = () => {
      const time = clock.getElapsedTime();
      stars.rotation.y = time * 0.05;
      stars.rotation.x = time * 0.03;
      
      stars.geometry.attributes.size.array.forEach((s, i) => {
        stars.geometry.attributes.size.array[i] = s * (1 + Math.sin(time + i) * 0.2);
      });
      stars.geometry.attributes.size.needsUpdate = true;
    };

    // Настройка управления камерой
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 1.5;
    controls.zoomSpeed = 2;
    controls.panSpeed = 1.5;
    controls.minDistance = 30;
    controls.maxDistance = 500;
    controls.enableKeys = true;
    
    // Анимационный цикл
    const animate = () => {
      requestAnimationFrame(animate);
      animateStars();
      controls.update();
      composer.render();
    };
    animate();

    // Обработка изменения размеров
    const handleResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      composer.setSize(w, h);
    };
    
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      container.removeChild(renderer.domElement);
    };
  }, [starColors]);

  return (
    <div
      ref={mountRef}
      style={{
        width: "100%",
        height: "100vh",
        position: "fixed",
        top: 0,
        left: 0,
        zIndex: 0,
        overflow: "hidden"
      }}
    />
  );
};

export default ThreeScene;
