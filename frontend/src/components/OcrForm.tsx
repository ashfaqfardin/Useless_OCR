"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Upload,
  FileText,
  Copy,
  Download,
  Loader2,
  X,
  Zap,
  Eye,
  Edit3,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Maximize,
  Minimize,
  Move,
  MousePointer,
  Square,
  Info,
  Save,
} from "lucide-react";
import { toast } from "sonner";

interface OcrResult {
  results: Array<{
    bbox: number[][];
    text: string;
    confidence: number;
  }>;
}

export function OcrForm() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<OcrResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [selectedBox, setSelectedBox] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [isCreatingBox, setIsCreatingBox] = useState(false);
  const [newBoxStart, setNewBoxStart] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [newBoxEnd, setNewBoxEnd] = useState<{ x: number; y: number } | null>(
    null
  );
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [rotation, setRotation] = useState(0);
  const [showInfo, setShowInfo] = useState(false);
  const [toolMode, setToolMode] = useState<"select" | "pan" | "create">(
    "select"
  );
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredBox, setHoveredBox] = useState<number | null>(null);
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(["en"]);
  const [useGroq, setUseGroq] = useState(false);
  const [useHybrid, setUseHybrid] = useState(false);
  const [groqApiKey, setGroqApiKey] = useState("");
  const [selectedGroqModel, setSelectedGroqModel] = useState(
    "meta-llama/llama-4-maverick-17b-128e-instruct"
  );
  const [groqModels, setGroqModels] = useState<
    Array<{ id: string; name: string; description: string; max_tokens: number }>
  >([]);
  const [customPrompt, setCustomPrompt] = useState("");
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please select an image file (PNG, JPG, JPEG, WebP)");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      // 10MB limit
      toast.error("Please select an image smaller than 10MB");
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      const files = e.dataTransfer.files;
      if (files && files[0]) {
        handleFileSelect(files[0]);
      }
    },
    [handleFileSelect]
  );

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Load available Groq models
  const loadGroqModels = useCallback(async () => {
    setIsLoadingModels(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/groq-models");
      if (response.ok) {
        const data = await response.json();
        setGroqModels(data.models);
      } else {
        // Fallback to hardcoded models if backend is not available
        const fallbackModels = [
          {
            id: "meta-llama/llama-4-maverick-17b-128e-instruct",
            name: "Llama 4 Maverick 17B 128E (Vision)",
            description:
              "Multimodal (text + image) - Use for OCR and image tasks",
            max_tokens: 8192,
          },
          {
            id: "llama3-8b-8192",
            name: "Llama 3 8B",
            description: "Fast and efficient 8B parameter model (text only)",
            max_tokens: 8192,
          },
          {
            id: "llama3-70b-8192",
            name: "Llama 3 70B",
            description: "High quality 70B parameter model (text only)",
            max_tokens: 8192,
          },
          {
            id: "mixtral-8x7b-32768",
            name: "Mixtral 8x7B",
            description:
              "High performance mixture of experts model (text only)",
            max_tokens: 32768,
          },
          {
            id: "gemma2-9b-it",
            name: "Gemma 2 9B",
            description: "Google's efficient 9B parameter model (text only)",
            max_tokens: 8192,
          },
        ];
        setGroqModels(fallbackModels);
      }
    } catch (error) {
      console.error(
        "Failed to load Groq models from backend, using fallback:",
        error
      );
      // Fallback to hardcoded models if backend is not available
      const fallbackModels = [
        {
          id: "meta-llama/llama-4-maverick-17b-128e-instruct",
          name: "Llama 4 Maverick 17B 128E (Vision)",
          description:
            "Multimodal (text + image) - Use for OCR and image tasks",
          max_tokens: 8192,
        },
        {
          id: "llama3-8b-8192",
          name: "Llama 3 8B",
          description: "Fast and efficient 8B parameter model",
          max_tokens: 8192,
        },
        {
          id: "llama3-70b-8192",
          name: "Llama 3 70B",
          description: "High quality 70B parameter model",
          max_tokens: 8192,
        },
        {
          id: "mixtral-8x7b-32768",
          name: "Mixtral 8x7B",
          description: "High performance mixture of experts model",
          max_tokens: 32768,
        },
        {
          id: "gemma2-9b-it",
          name: "Gemma 2 9B",
          description: "Google's efficient 9B parameter model",
          max_tokens: 8192,
        },
      ];
      setGroqModels(fallbackModels);
    } finally {
      setIsLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    if (useGroq || useHybrid) {
      loadGroqModels();
    }
  }, [useGroq, useHybrid, loadGroqModels]);

  const processOcr = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      if (useHybrid) {
        // Use Hybrid OCR (Groq + EasyOCR)
        if (!groqApiKey.trim()) {
          toast.error("Please enter your Groq API key");
          setIsProcessing(false);
          return;
        }

        formData.append("groq_api_key", groqApiKey);
        formData.append("model", selectedGroqModel);
        formData.append("languages", selectedLanguages.join(","));
        if (customPrompt.trim()) {
          formData.append("prompt", customPrompt);
        }

        const progressInterval = setInterval(() => {
          setProgress((prev) => Math.min(prev + 10, 90));
        }, 400);

        const response = await fetch("http://127.0.0.1:8000/hybrid-ocr", {
          method: "POST",
          body: formData,
        });

        clearInterval(progressInterval);
        setProgress(100);

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(
            errorData.error || `HTTP error! status: ${response.status}`
          );
        }

        const data = await response.json();

        // Transform hybrid response to match our interface
        const transformedResult: OcrResult = {
          results: data.results.map(
            (item: { bbox: number[][]; text: string; confidence: number }) => ({
              bbox: item.bbox,
              text: item.text,
              confidence: item.confidence,
            })
          ),
        };

        setResult(transformedResult);
        toast.success(
          `Hybrid OCR completed using ${data.model_used} + EasyOCR`
        );
      } else if (useGroq) {
        // Use Groq API only
        if (!groqApiKey.trim()) {
          toast.error("Please enter your Groq API key");
          setIsProcessing(false);
          return;
        }

        formData.append("groq_api_key", groqApiKey);
        formData.append("model", selectedGroqModel);
        if (customPrompt.trim()) {
          formData.append("prompt", customPrompt);
        }

        const progressInterval = setInterval(() => {
          setProgress((prev) => Math.min(prev + 15, 90));
        }, 300);

        const response = await fetch("http://127.0.0.1:8000/groq-ocr", {
          method: "POST",
          body: formData,
        });

        clearInterval(progressInterval);
        setProgress(100);

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(
            errorData.error || `HTTP error! status: ${response.status}`
          );
        }

        const data = await response.json();

        // Transform Groq response to match our interface
        const transformedResult: OcrResult = {
          results: data.results.map(
            (item: { bbox: number[][]; text: string; confidence: number }) => ({
              bbox: item.bbox,
              text: item.text,
              confidence: item.confidence,
            })
          ),
        };

        setResult(transformedResult);
        toast.success(`Text extraction completed using ${data.model_used}`);
      } else {
        // Use EasyOCR only
        formData.append("languages", selectedLanguages.join(","));

        const progressInterval = setInterval(() => {
          setProgress((prev) => Math.min(prev + 20, 90));
        }, 500);

        const response = await fetch("http://127.0.0.1:8000/ocr", {
          method: "POST",
          body: formData,
        });

        clearInterval(progressInterval);
        setProgress(100);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Transform backend response to match our interface
        const transformedResult: OcrResult = {
          results: data.results.map(
            (item: { bbox: number[][]; text: string; confidence: number }) => ({
              bbox: item.bbox,
              text: item.text,
              confidence: item.confidence,
            })
          ),
        };

        setResult(transformedResult);
        toast.success("Text extraction completed successfully");
      }
    } catch (error) {
      console.error("OCR processing error:", error);
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to process image. Please try again."
      );
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  };

  const copyToClipboard = async () => {
    if (result?.results) {
      await navigator.clipboard.writeText(
        result.results.map((r) => r.text).join("\n")
      );
      toast.success("Text has been copied to your clipboard");
    }
  };

  const downloadText = () => {
    if (result?.results) {
      const blob = new Blob([result.results.map((r) => r.text).join("\n")], {
        type: "text/plain",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "extracted-text.txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const clearAll = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setProgress(0);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  };

  // Handle wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom((prev) => Math.max(0.1, Math.min(10, prev * delta)));
    }
  }, []);

  // Enhanced zoom controls
  const zoomIn = useCallback(
    () => setZoom((prev) => Math.min(prev * 1.5, 10)),
    []
  );
  const zoomOut = useCallback(
    () => setZoom((prev) => Math.max(prev / 1.5, 0.1)),
    []
  );
  const zoomToFit = useCallback(() => {
    if (!imageRef.current || !containerRef.current) return;
    const container = containerRef.current.getBoundingClientRect();
    const image = imageRef.current.getBoundingClientRect();
    const scaleX = container.width / image.width;
    const scaleY = container.height / image.height;
    const scale = Math.min(scaleX, scaleY, 1);
    setZoom(scale);
    setPan({ x: 0, y: 0 });
  }, []);
  const zoomToActual = useCallback(() => setZoom(1), []);
  const rotateImage = useCallback(
    () => setRotation((prev) => (prev + 90) % 360),
    []
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;

      switch (e.key) {
        case "Escape":
          setSelectedBox(null);
          setIsEditing(false);
          break;
        case "Delete":
        case "Backspace":
          if (selectedBox !== null && result) {
            const newResults = result.results.filter(
              (_, index) => index !== selectedBox
            );
            setResult({ ...result, results: newResults });
            setSelectedBox(null);
            toast.success("Region deleted");
          }
          break;
        case "1":
          setToolMode("select");
          break;
        case "2":
          setToolMode("pan");
          break;
        case "3":
          setToolMode("create");
          break;
        case "0":
          zoomToFit();
          break;
        case "=":
        case "+":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomIn();
          }
          break;
        case "-":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomOut();
          }
          break;
        case "r":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            rotateImage();
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedBox, result, zoomToFit, zoomIn, zoomOut, rotateImage]);

  // Helper function to get image coordinates
  const getImageCoordinates = useCallback(
    (e: React.MouseEvent) => {
      if (!imageRef.current || !containerRef.current) return null;

      const imageRect = imageRef.current.getBoundingClientRect();
      const containerRect = containerRef.current.getBoundingClientRect();

      // Calculate mouse position relative to the container
      const containerX = e.clientX - containerRect.left;
      const containerY = e.clientY - containerRect.top;

      // Calculate image position within container
      const imageLeft = imageRect.left - containerRect.left;
      const imageTop = imageRect.top - containerRect.top;

      // Calculate mouse position relative to the image, accounting for zoom and pan
      const imageX = (containerX - imageLeft - pan.x) / zoom;
      const imageY = (containerY - imageTop - pan.y) / zoom;

      // Check if click is within image bounds
      if (
        imageX < 0 ||
        imageX > imageRect.width ||
        imageY < 0 ||
        imageY > imageRect.height
      ) {
        return null;
      }

      // Scale coordinates to match the original image dimensions
      const scaleX = imageRef.current.naturalWidth / imageRect.width;
      const scaleY = imageRef.current.naturalHeight / imageRect.height;

      return {
        x: imageX * scaleX,
        y: imageY * scaleY,
      };
    },
    [zoom, pan]
  );

  // Helper: calculate distance from point to box edge
  const getDistanceToBox = useCallback(
    (x: number, y: number, bbox: number[][]) => {
      // Calculate minimum distance from point to any edge of the polygon
      let minDistance = Infinity;

      for (let i = 0; i < bbox.length; i++) {
        const p1 = bbox[i];
        const p2 = bbox[(i + 1) % bbox.length];

        // Distance from point to line segment
        const A = x - p1[0];
        const B = y - p1[1];
        const C = p2[0] - p1[0];
        const D = p2[1] - p1[1];
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;

        if (lenSq !== 0) param = dot / lenSq;

        let xx, yy;
        if (param < 0) {
          xx = p1[0];
          yy = p1[1];
        } else if (param > 1) {
          xx = p2[0];
          yy = p2[1];
        } else {
          xx = p1[0] + param * C;
          yy = p1[1] + param * D;
        }

        const dx = x - xx;
        const dy = y - yy;
        const distance = Math.sqrt(dx * dx + dy * dy);
        minDistance = Math.min(minDistance, distance);
      }

      return minDistance;
    },
    []
  );

  // Helper: check if point is near the bounding box (buffer in px)
  const isPointNearBox = useCallback(
    (x: number, y: number, bbox: number[][], buffer = 8) => {
      const minX = Math.min(...bbox.map((p) => p[0])) - buffer;
      const maxX = Math.max(...bbox.map((p) => p[0])) + buffer;
      const minY = Math.min(...bbox.map((p) => p[1])) - buffer;
      const maxY = Math.max(...bbox.map((p) => p[1])) + buffer;
      return x >= minX && x <= maxX && y >= minY && y <= maxY;
    },
    []
  );

  // Check if point is inside polygon
  const isPointInPolygon = useCallback(
    (x: number, y: number, polygon: number[][]) => {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        if (
          polygon[i][1] > y !== polygon[j][1] > y &&
          x <
            ((polygon[j][0] - polygon[i][0]) * (y - polygon[i][1])) /
              (polygon[j][1] - polygon[i][1]) +
              polygon[i][0]
        ) {
          inside = !inside;
        }
      }
      return inside;
    },
    []
  );

  // Update text for a specific region
  const updateText = useCallback(
    (index: number, newText: string) => {
      if (!result) return;

      const newResult = { ...result };
      newResult.results[index].text = newText;
      setResult(newResult);
    },
    [result]
  );

  // Draw bounding boxes on canvas with editing capabilities
  const drawBoundingBoxes = useCallback(() => {
    if (!canvasRef.current || !imageRef.current || !result) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Get container (canvas) size
    const container = canvas.parentElement;
    const containerWidth = container ? container.clientWidth : canvas.width;
    const containerHeight = container ? container.clientHeight : canvas.height;
    canvas.width = containerWidth;
    canvas.height = containerHeight;

    // Get image natural size
    const image = imageRef.current;
    const naturalWidth = image.naturalWidth;
    const naturalHeight = image.naturalHeight;

    // Calculate displayed image size and offset (object-contain)
    const containerAspect = containerWidth / containerHeight;
    const imageAspect = naturalWidth / naturalHeight;
    let displayedWidth, displayedHeight, offsetX, offsetY;
    if (imageAspect > containerAspect) {
      // Image fills width, letterbox top/bottom
      displayedWidth = containerWidth;
      displayedHeight = containerWidth / imageAspect;
      offsetX = 0;
      offsetY = (containerHeight - displayedHeight) / 2;
    } else {
      // Image fills height, letterbox left/right
      displayedHeight = containerHeight;
      displayedWidth = containerHeight * imageAspect;
      offsetX = (containerWidth - displayedWidth) / 2;
      offsetY = 0;
    }

    // Calculate scale factors from original image to displayed image
    const scaleX = displayedWidth / naturalWidth;
    const scaleY = displayedHeight / naturalHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bounding boxes
    result.results.forEach((item, index) => {
      const bbox = item.bbox;
      const isSelected = selectedBox === index;
      const isHovered = hoveredBox === index;

      // Scale and offset the bounding box coordinates
      const scaledBbox = bbox.map((point) => [
        point[0] * scaleX + offsetX,
        point[1] * scaleY + offsetY,
      ]);

      // Choose color based on state
      let strokeColor = `hsl(${(index * 137.5) % 360}, 70%, 50%)`;
      let lineWidth = 2;

      if (isSelected) {
        strokeColor = "#ff6b6b";
        lineWidth = 3;
      } else if (isHovered && toolMode === "select" && !isEditing) {
        strokeColor = "#3b82f6";
        lineWidth = 2.5;
      }

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(scaledBbox[0][0], scaledBbox[0][1]);
      ctx.lineTo(scaledBbox[1][0], scaledBbox[1][1]);
      ctx.lineTo(scaledBbox[2][0], scaledBbox[2][1]);
      ctx.lineTo(scaledBbox[3][0], scaledBbox[3][1]);
      ctx.closePath();
      ctx.stroke();

      if (isSelected) {
        ctx.fillStyle = "#ff6b6b";
        const handleSize = 6;
        scaledBbox.forEach((point) => {
          ctx.fillRect(
            point[0] - handleSize / 2,
            point[1] - handleSize / 2,
            handleSize,
            handleSize
          );
        });
      }
    });
    // Draw new box preview if creating
    if (isCreatingBox && newBoxStart && newBoxEnd) {
      const startX = newBoxStart.x * scaleX + offsetX;
      const startY = newBoxStart.y * scaleY + offsetY;
      const endX = newBoxEnd.x * scaleX + offsetX;
      const endY = newBoxEnd.y * scaleY + offsetY;
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.rect(
        Math.min(startX, endX),
        Math.min(startY, endY),
        Math.abs(endX - startX),
        Math.abs(endY - startY)
      );
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [
    result,
    selectedBox,
    hoveredBox,
    toolMode,
    isEditing,
    isCreatingBox,
    newBoxStart,
    newBoxEnd,
  ]);

  useEffect(() => {
    drawBoundingBoxes();
  }, [drawBoundingBoxes, zoom, pan]);

  // Handle image load to ensure bounding boxes are drawn
  useEffect(() => {
    const img = imageRef.current;
    const handleImageLoad = () => {
      drawBoundingBoxes();
    };

    if (img) {
      img.addEventListener("load", handleImageLoad);
    }

    return () => {
      if (img) {
        img.removeEventListener("load", handleImageLoad);
      }
    };
  }, [drawBoundingBoxes]);

  // Redraw bounding boxes on window resize and image load
  useEffect(() => {
    const handleResize = () => {
      drawBoundingBoxes();
    };

    // Create a ResizeObserver to watch for image size changes
    const resizeObserver = new ResizeObserver(() => {
      drawBoundingBoxes();
    });

    // Observe the image element for size changes
    if (imageRef.current) {
      resizeObserver.observe(imageRef.current);
    }

    // Observe the container for size changes
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      resizeObserver.disconnect();
    };
  }, [drawBoundingBoxes]);

  // Restore exportAnnotations function
  const exportAnnotations = useCallback(() => {
    if (!result || !selectedFile) return;

    const annotations = {
      filename: selectedFile.name,
      timestamp: new Date().toISOString(),
      imageUrl: previewUrl,
      annotations: result.results.map((item, index) => ({
        id: index,
        bbox: item.bbox,
        text: item.text,
        confidence: item.confidence,
        region: index + 1,
      })),
    };

    const blob = new Blob([JSON.stringify(annotations, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedFile.name.replace(
      /\.[^/.]+$/,
      ""
    )}_annotations.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast.success("Annotations exported successfully!");
  }, [result, selectedFile, previewUrl]);

  // Enhanced mouse interactions based on tool mode
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!result || !containerRef.current || !imageRef.current) return;

      // Handle panning based on tool mode
      if (
        toolMode === "pan" ||
        e.button === 1 ||
        (e.button === 0 && e.altKey)
      ) {
        e.preventDefault();
        setIsPanning(true);
        setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
        return;
      }

      // Handle box creation
      if (toolMode === "create" && e.button === 0) {
        const coords = getImageCoordinates(e);
        if (coords) {
          setSelectedBox(null);
          setIsCreatingBox(true);
          setNewBoxStart(coords);
          setNewBoxEnd(coords);
        }
        return;
      }

      // Handle box selection (default tool mode)
      if (toolMode === "select" && e.button === 0) {
        const coords = getImageCoordinates(e);
        if (!coords) return;

        // Find all boxes that are near the click point
        const nearbyBoxes: Array<{ index: number; distance: number }> = [];

        result.results.forEach((item, index) => {
          const bbox = item.bbox;
          if (
            isPointInPolygon(coords.x, coords.y, bbox) ||
            isPointNearBox(coords.x, coords.y, bbox, 8)
          ) {
            const distance = getDistanceToBox(coords.x, coords.y, bbox);
            nearbyBoxes.push({ index, distance });
          }
        });

        // Select the closest box
        if (nearbyBoxes.length > 0) {
          const closestBox = nearbyBoxes.reduce((closest, current) =>
            current.distance < closest.distance ? current : closest
          );

          setSelectedBox(closestBox.index);
          if (isEditing) {
            const bbox = result.results[closestBox.index].bbox;
            setIsDragging(true);
            setDragOffset({
              x: coords.x - bbox[0][0],
              y: coords.y - bbox[0][1],
            });
          }
        } else if (isEditing) {
          // If not clicking on a box and in edit mode, start creating a new one
          setSelectedBox(null);
          setIsCreatingBox(true);
          setNewBoxStart(coords);
          setNewBoxEnd(coords);
        }
      }
    },
    [
      toolMode,
      isEditing,
      result,
      getImageCoordinates,
      isPointInPolygon,
      isPointNearBox,
      getDistanceToBox,
      pan,
    ]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      // Handle panning
      if (isPanning) {
        setPan({
          x: e.clientX - panStart.x,
          y: e.clientY - panStart.y,
        });
        return;
      }

      // Handle dragging
      if (isDragging && selectedBox !== null && result) {
        const coords = getImageCoordinates(e);
        if (!coords) return;

        const newResult = { ...result };
        const bbox = newResult.results[selectedBox].bbox;
        const width = bbox[1][0] - bbox[0][0];
        const height = bbox[2][1] - bbox[1][1];

        // Update bounding box position
        newResult.results[selectedBox].bbox = [
          [coords.x - dragOffset.x, coords.y - dragOffset.y],
          [coords.x - dragOffset.x + width, coords.y - dragOffset.y],
          [coords.x - dragOffset.x + width, coords.y - dragOffset.y + height],
          [coords.x - dragOffset.x, coords.y - dragOffset.y + height],
        ];

        setResult(newResult);
      } else if (isCreatingBox && newBoxStart) {
        // Handle creating new box
        const coords = getImageCoordinates(e);
        if (coords) {
          setNewBoxEnd(coords);
        }
      } else if (toolMode === "select" && !isEditing) {
        // Handle hover for box highlighting
        const coords = getImageCoordinates(e);
        if (coords && result) {
          let closestBox: number | null = null;
          let minDistance = Infinity;

          result.results.forEach((item, index) => {
            const bbox = item.bbox;
            if (isPointNearBox(coords.x, coords.y, bbox, 8)) {
              const distance = getDistanceToBox(coords.x, coords.y, bbox);
              if (distance < minDistance) {
                minDistance = distance;
                closestBox = index;
              }
            }
          });

          setHoveredBox(closestBox);
        } else {
          setHoveredBox(null);
        }
      }
    },
    [
      isPanning,
      panStart,
      isDragging,
      isCreatingBox,
      selectedBox,
      result,
      dragOffset,
      newBoxStart,
      isEditing,
      toolMode,
      getImageCoordinates,
      isPointNearBox,
      getDistanceToBox,
    ]
  );

  const handleMouseUp = useCallback(() => {
    // Handle panning end
    if (isPanning) {
      setIsPanning(false);
      return;
    }

    if (isCreatingBox && newBoxStart && newBoxEnd && result) {
      // Create new bounding box
      const newResult: OcrResult = { ...result };
      const newBox = {
        bbox: [
          [
            Math.min(newBoxStart.x, newBoxEnd.x),
            Math.min(newBoxStart.y, newBoxEnd.y),
          ],
          [
            Math.max(newBoxStart.x, newBoxEnd.x),
            Math.min(newBoxStart.y, newBoxEnd.y),
          ],
          [
            Math.max(newBoxStart.x, newBoxEnd.x),
            Math.max(newBoxStart.y, newBoxEnd.y),
          ],
          [
            Math.min(newBoxStart.x, newBoxEnd.x),
            Math.max(newBoxStart.y, newBoxEnd.y),
          ],
        ],
        text: "New text region",
        confidence: 0.0,
      };

      newResult.results.push(newBox);
      setResult(newResult);
      setSelectedBox(newResult.results.length - 1);
    }

    setIsDragging(false);
    setIsCreatingBox(false);
    setNewBoxStart(null);
    setNewBoxEnd(null);
  }, [isPanning, isCreatingBox, newBoxStart, newBoxEnd, result]);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <div className="container mx-auto px-6 py-12 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-3xl font-semibold text-slate-900 dark:text-slate-100 mb-3">
            OCR Text Extractor
          </h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-lg mx-auto">
            Extract text from images with AI-powered OCR
          </p>
        </div>

        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full grid-cols-3 max-w-sm mx-auto mb-8 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
            <TabsTrigger
              value="upload"
              className="flex items-center gap-2 data-[state=active]:bg-slate-900 data-[state=active]:text-white dark:data-[state=active]:bg-white dark:data-[state=active]:text-slate-900 transition-colors"
            >
              <Upload className="h-4 w-4" />
              <span className="hidden sm:inline">Upload</span>
              <span className="sm:hidden">Upload</span>
            </TabsTrigger>
            <TabsTrigger
              value="result"
              className="flex items-center gap-2 data-[state=active]:bg-slate-900 data-[state=active]:text-white dark:data-[state=active]:bg-white dark:data-[state=active]:text-slate-900 transition-colors"
              disabled={!result}
            >
              <FileText className="h-4 w-4" />
              <span className="hidden sm:inline">Results</span>
              <span className="sm:hidden">Results</span>
            </TabsTrigger>
            <TabsTrigger
              value="visualize"
              className="flex items-center gap-2 data-[state=active]:bg-slate-900 data-[state=active]:text-white dark:data-[state=active]:bg-white dark:data-[state=active]:text-slate-900 transition-colors"
              disabled={!result}
            >
              <Eye className="h-4 w-4" />
              <span className="hidden sm:inline">Visualize</span>
              <span className="sm:hidden">Visualize</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            {/* Upload Section */}
            <Card className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
              <CardHeader>
                <CardTitle className="text-xl">Upload Image</CardTitle>
                <p className="text-slate-600 dark:text-slate-400">
                  Select an image file to extract text using OCR
                </p>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* OCR Method Selection */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    OCR Method
                  </label>
                  <div className="flex items-center space-x-4">
                    <label className="flex items-center space-x-2">
                      <input
                        type="radio"
                        checked={!useGroq && !useHybrid}
                        onChange={() => {
                          setUseGroq(false);
                          setUseHybrid(false);
                        }}
                        className="rounded"
                      />
                      <span className="text-sm">EasyOCR (Local)</span>
                    </label>
                    <label className="flex items-center space-x-2">
                      <input
                        type="radio"
                        checked={useGroq && !useHybrid}
                        onChange={() => {
                          setUseGroq(true);
                          setUseHybrid(false);
                        }}
                        className="rounded"
                      />
                      <span className="text-sm">Groq API (Cloud)</span>
                    </label>
                    <label className="flex items-center space-x-2">
                      <input
                        type="radio"
                        checked={useHybrid}
                        onChange={() => {
                          setUseGroq(false);
                          setUseHybrid(true);
                        }}
                        className="rounded"
                      />
                      <span className="text-sm">Hybrid (Groq + EasyOCR)</span>
                    </label>
                  </div>
                </div>

                {/* EasyOCR Language Selection */}
                {!useGroq || useHybrid ? (
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      Languages to Detect
                    </label>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                      {[
                        { code: "en", name: "English" },
                        { code: "bn", name: "বাংলা (Bangla)" },
                        { code: "fr", name: "Français" },
                        { code: "de", name: "Deutsch" },
                        { code: "es", name: "Español" },
                        { code: "hi", name: "हिन्दी" },
                        { code: "ar", name: "العربية" },
                        { code: "zh", name: "中文" },
                        { code: "ja", name: "日本語" },
                        { code: "ko", name: "한국어" },
                        { code: "ru", name: "Русский" },
                        { code: "pt", name: "Português" },
                      ].map((lang) => (
                        <label
                          key={lang.code}
                          className="flex items-center space-x-2 p-2 rounded border cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-700"
                        >
                          <input
                            type="checkbox"
                            checked={selectedLanguages.includes(lang.code)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedLanguages([
                                  ...selectedLanguages,
                                  lang.code,
                                ]);
                              } else {
                                setSelectedLanguages(
                                  selectedLanguages.filter(
                                    (l) => l !== lang.code
                                  )
                                );
                              }
                            }}
                            className="rounded"
                          />
                          <span className="text-sm">{lang.name}</span>
                        </label>
                      ))}
                    </div>
                    {selectedLanguages.length === 0 && (
                      <p className="text-sm text-red-500">
                        Please select at least one language
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      Model
                    </label>
                    <select
                      value={selectedGroqModel}
                      onChange={(e) => setSelectedGroqModel(e.target.value)}
                      className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      disabled={isLoadingModels}
                    >
                      {groqModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name} - {model.description}
                        </option>
                      ))}
                    </select>
                    {selectedGroqModel !==
                      "meta-llama/llama-4-maverick-17b-128e-instruct" && (
                      <p className="text-xs text-red-500 mt-1">
                        Only <b>Llama 4 Maverick 17B 128E (Vision)</b> supports
                        image input. Other models will ignore the image and only
                        process text prompts.
                      </p>
                    )}
                  </div>
                )}

                {/* Groq API Configuration */}
                {(useGroq || useHybrid) && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        Groq API Key
                      </label>
                      <input
                        type="password"
                        value={groqApiKey}
                        onChange={(e) => setGroqApiKey(e.target.value)}
                        placeholder="Enter your Groq API key"
                        className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                      <p className="text-xs text-slate-500">
                        Get your API key from{" "}
                        <a
                          href="https://console.groq.com/"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline"
                        >
                          Groq Console
                        </a>
                      </p>
                    </div>

                    {useGroq && (
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                          Custom Prompt (Optional)
                        </label>
                        <textarea
                          value={customPrompt}
                          onChange={(e) => setCustomPrompt(e.target.value)}
                          placeholder="Enter a custom prompt for text extraction..."
                          rows={3}
                          className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                        <p className="text-xs text-slate-500">
                          Leave empty to use the default text extraction prompt
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* File Upload */}
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                    dragActive
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20"
                      : "border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  {previewUrl ? (
                    <div className="space-y-4">
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="max-h-64 mx-auto rounded-lg shadow-sm"
                      />
                      <div className="flex items-center justify-center gap-2">
                        <Button
                          onClick={processOcr}
                          disabled={
                            isProcessing ||
                            (!useGroq &&
                              !useHybrid &&
                              selectedLanguages.length === 0) ||
                            ((useGroq || useHybrid) && !groqApiKey.trim())
                          }
                        >
                          {isProcessing ? (
                            <>
                              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              Processing...
                            </>
                          ) : (
                            <>
                              <Zap className="h-4 w-4 mr-2" />
                              Extract Text
                            </>
                          )}
                        </Button>
                        <Button variant="outline" onClick={clearAll}>
                          <X className="h-4 w-4 mr-2" />
                          Clear
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Upload className="h-12 w-12 mx-auto text-slate-400" />
                      <div>
                        <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
                          Drop your image here
                        </p>
                        <p className="text-slate-600 dark:text-slate-400">
                          or click to browse files
                        </p>
                      </div>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileInputChange}
                        className="hidden"
                        id="file-upload"
                      />
                      <label
                        htmlFor="file-upload"
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700"
                      >
                        Choose File
                      </label>
                    </div>
                  )}
                </div>

                {/* Progress Bar */}
                {isProcessing && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm text-slate-600 dark:text-slate-400">
                      <span>Processing image...</span>
                      <span>{progress}%</span>
                    </div>
                    <Progress value={progress} className="w-full" />
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="result" className="space-y-6">
            {result && (
              <>
                {/* Results Summary */}
                <Card className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                  <CardContent className="p-6">
                    <div className="grid grid-cols-3 gap-6 text-center">
                      <div>
                        <div className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
                          {result.results.length}
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                          Text Regions
                        </div>
                      </div>
                      <div>
                        <div className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
                          {(
                            (result.results.reduce(
                              (sum, r) => sum + r.confidence,
                              0
                            ) /
                              result.results.length) *
                            100
                          ).toFixed(1)}
                          %
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                          Avg Confidence
                        </div>
                      </div>
                      <div>
                        <div className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
                          {result.results.reduce(
                            (sum, r) => sum + r.text.length,
                            0
                          )}
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                          Characters
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Extracted Text */}
                <Card className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                  <CardHeader className="border-b border-slate-200 dark:border-slate-700">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Extracted Text</CardTitle>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={copyToClipboard}
                          className="flex items-center gap-2"
                        >
                          <Copy className="h-4 w-4" />
                          Copy
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={downloadText}
                          className="flex items-center gap-2"
                        >
                          <Download className="h-4 w-4" />
                          Download
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-6">
                    <Textarea
                      value={result.results.map((r) => r.text).join("\n")}
                      readOnly
                      className="min-h-[400px] resize-none text-base leading-relaxed border-0 bg-slate-50 dark:bg-slate-900 focus-visible:ring-0 focus-visible:ring-offset-0"
                      placeholder="Extracted text will appear here..."
                    />
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          <TabsContent value="visualize" className="space-y-6">
            {result && previewUrl && (
              <>
                {/* Enhanced Toolbar */}
                <Card className="border-0 shadow-sm bg-background/50 backdrop-blur">
                  <CardContent className="p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {/* Tool Selection */}
                        <div className="flex items-center gap-1 bg-muted rounded-md p-1">
                          <Button
                            variant={
                              toolMode === "select" ? "default" : "ghost"
                            }
                            size="sm"
                            onClick={() => setToolMode("select")}
                            className="h-8 w-8 p-0"
                          >
                            <MousePointer className="h-4 w-4" />
                          </Button>
                          <Button
                            variant={toolMode === "pan" ? "default" : "ghost"}
                            size="sm"
                            onClick={() => setToolMode("pan")}
                            className="h-8 w-8 p-0"
                          >
                            <Move className="h-4 w-4" />
                          </Button>
                          <Button
                            variant={
                              toolMode === "create" ? "default" : "ghost"
                            }
                            size="sm"
                            onClick={() => setToolMode("create")}
                            className="h-8 w-8 p-0"
                          >
                            <Square className="h-4 w-4" />
                          </Button>
                        </div>

                        <div className="w-px h-6 bg-border mx-2" />

                        {/* Zoom Controls */}
                        <div className="flex items-center gap-1">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={zoomOut}
                            className="h-8 w-8 p-0"
                          >
                            <ZoomOut className="h-4 w-4" />
                          </Button>
                          <div className="flex items-center gap-1 bg-muted rounded-md px-2 py-1 min-w-[80px] justify-center">
                            <span className="text-xs font-medium">
                              {Math.round(zoom * 100)}%
                            </span>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={zoomIn}
                            className="h-8 w-8 p-0"
                          >
                            <ZoomIn className="h-4 w-4" />
                          </Button>
                        </div>

                        <div className="w-px h-6 bg-border mx-2" />

                        {/* Fit Controls */}
                        <div className="flex items-center gap-1">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={zoomToFit}
                            className="h-8 px-3 text-xs"
                          >
                            <Maximize className="h-4 w-4 mr-1" />
                            Fit
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={zoomToActual}
                            className="h-8 px-3 text-xs"
                          >
                            <Minimize className="h-4 w-4 mr-1" />
                            100%
                          </Button>
                        </div>

                        <div className="w-px h-6 bg-border mx-2" />

                        {/* Rotation */}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={rotateImage}
                          className="h-8 w-8 p-0"
                        >
                          <RotateCw className="h-4 w-4" />
                        </Button>
                      </div>

                      <div className="flex items-center gap-2">
                        {/* Info Panel Toggle */}
                        <Button
                          variant={showInfo ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowInfo(!showInfo)}
                          className="h-8 w-8 p-0"
                        >
                          <Info className="h-4 w-4" />
                        </Button>

                        {/* Edit Mode Toggle */}
                        <Button
                          variant={isEditing ? "default" : "outline"}
                          size="sm"
                          onClick={() => setIsEditing(!isEditing)}
                          className="h-8 px-3 text-xs"
                        >
                          <Edit3 className="h-4 w-4 mr-1" />
                          {isEditing ? "View" : "Edit"}
                        </Button>

                        {/* Export JSON Button */}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={exportAnnotations}
                          className="h-8 px-3 text-xs"
                          disabled={!result || !selectedFile}
                        >
                          <Save className="h-4 w-4 mr-1" />
                          Export JSON
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Info Panel */}
                {showInfo && previewUrl && (
                  <Card className="border-0 shadow-sm bg-background/50 backdrop-blur">
                    <CardContent className="p-4">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-muted-foreground">
                            Image Size
                          </div>
                          <div className="font-medium">
                            {imageRef.current
                              ? `${imageRef.current.naturalWidth} × ${imageRef.current.naturalHeight}`
                              : "Loading..."}
                          </div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">
                            Display Size
                          </div>
                          <div className="font-medium">
                            {imageRef.current
                              ? `${Math.round(
                                  imageRef.current.width
                                )} × ${Math.round(imageRef.current.height)}`
                              : "Loading..."}
                          </div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">
                            Zoom Level
                          </div>
                          <div className="font-medium">
                            {Math.round(zoom * 100)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">
                            Text Regions
                          </div>
                          <div className="font-medium">
                            {result?.results?.length || 0}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                <div className="h-[calc(100vh-300px)]">
                  {/* Image Viewer */}
                  <Card className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex flex-col h-full">
                    <CardHeader className="border-b border-slate-200 dark:border-slate-700 flex-shrink-0">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">Image Viewer</CardTitle>
                        <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
                          <span>{result.results.length} regions</span>
                          <span>
                            {(
                              (result.results.reduce(
                                (sum, r) => sum + r.confidence,
                                0
                              ) /
                                result.results.length) *
                              100
                            ).toFixed(1)}
                            % avg
                          </span>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-0 flex-1 overflow-hidden">
                      <div
                        ref={containerRef}
                        className={`relative w-full h-full bg-black ${
                          isEditing ? "cursor-crosshair" : "cursor-pointer"
                        }`}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onWheel={handleWheel}
                      >
                        <div
                          className="transform-wrapper absolute top-0 left-0 w-full h-full"
                          style={{
                            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom}) rotate(${rotation}deg)`,
                            transformOrigin: "center center",
                            transition: isPanning
                              ? "none"
                              : "transform 0.1s ease-out",
                            pointerEvents: "none",
                          }}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            ref={imageRef}
                            src={previewUrl}
                            alt="Preview with bounding boxes"
                            className={
                              isEditing
                                ? "max-h-full max-w-full object-contain"
                                : "max-h-full max-w-full object-contain"
                            }
                            onLoad={drawBoundingBoxes}
                            draggable={false}
                            style={{
                              display: "block",
                              width: "100%",
                              height: "100%",
                            }}
                          />
                          <canvas
                            ref={canvasRef}
                            className="absolute top-0 left-0 w-full h-full pointer-events-none"
                            style={{ width: "100%", height: "100%" }}
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Floating Text Panel */}
                  {selectedBox !== null && (
                    <div className="fixed top-4 right-4 w-80 max-h-96 bg-background border rounded-lg shadow-lg z-50">
                      <div className="p-4 border-b bg-muted/30">
                        <div className="flex items-center justify-between">
                          <h3 className="font-semibold">
                            Region {selectedBox + 1}
                          </h3>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setSelectedBox(null)}
                            className="h-6 w-6 p-0"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="p-4 max-h-64 overflow-y-auto">
                        {isEditing ? (
                          <Textarea
                            value={result.results[selectedBox].text}
                            onChange={(e) =>
                              updateText(selectedBox, e.target.value)
                            }
                            className="min-h-[100px] resize-none text-sm border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 p-0"
                            placeholder="Edit extracted text..."
                          />
                        ) : (
                          <p className="text-sm leading-relaxed text-foreground/90">
                            {result.results[selectedBox].text}
                          </p>
                        )}
                        <div className="mt-3 pt-3 border-t">
                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span>
                              Confidence:{" "}
                              {(
                                result.results[selectedBox].confidence * 100
                              ).toFixed(1)}
                              %
                            </span>
                            <span>Click to edit</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
