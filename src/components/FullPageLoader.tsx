"use client";

import React from "react";
import { Clock } from "lucide-react";

const FullPageLoader: React.FC = () => {
  return (
    <div className="flex items-center justify-center">
      <div className="flex items-center space-x-2">
        <Clock className="h-8 w-8 animate-spin text-blue-600" />
        <span className="text-lg font-medium text-gray-700">Loading...</span>
      </div>
    </div>
  );
};

export default FullPageLoader;
