/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {css} from 'lit';

/** Shared CSS styles for the Embedding Search app. */
export const sharedStyles = css`
  * {
    box-sizing: border-box;
  }

  h1, h2, h3, h4, p {
    margin: 0;
  }

  select {
    background-color: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    color: var(--text);
    font-size: 0.875rem;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
    font-family: inherit;
  }

  select:focus {
    border-color: var(--teal);
    box-shadow: 0 0 0 2px rgba(0, 201, 158, 0.2);
  }

  /* Buttons */
  .btn {
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-size: 0.875rem;
    transition: all 0.15s ease-in-out;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-family: inherit;
    user-select: none;
  }

  .btn-primary {
    background-color: var(--teal);
    color: #0b0f19;
  }

  .btn-primary:hover:not(:disabled) {
    background-color: #00b38c;
    transform: translateY(-1px);
  }

  .btn-secondary {
    background-color: var(--bg-card-hover);
    border: 1px solid var(--border);
    color: var(--text);
  }

  .btn-secondary:hover:not(:disabled) {
    background-color: #334155;
    border-color: var(--border-light);
  }

  .btn-danger {
    background-color: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
  }

  .btn-danger:hover:not(:disabled) {
    background-color: rgba(239, 68, 68, 0.3);
  }

  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none;
  }

  /* Badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1;
  }

  .badge-teal {
    background-color: rgba(0, 201, 158, 0.15);
    color: #00c99e;
    border: 1px solid rgba(0, 201, 158, 0.3);
  }

  .badge-blue {
    background-color: rgba(91, 227, 227, 0.15);
    color: var(--cyan);
    border: 1px solid rgba(91, 227, 227, 0.3);
  }

  .badge-purple {
    background-color: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
  }

  .badge-muted {
    background-color: rgba(100, 116, 139, 0.2);
    color: #94a3b8;
    border: 1px solid rgba(100, 116, 139, 0.3);
  }

  /* Modals */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background-color: rgba(11, 15, 25, 0.8);
    backdrop-filter: blur(5px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1100;
    padding: 20px;
  }

  .modal-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    width: 100%;
    max-width: 560px;
    box-shadow: 0 22px 40px -10px rgba(0, 0, 0, 0.65);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    color: #ffffff;
  }

  .modal-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
  }

  .close-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 1.2rem;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    line-height: 1;
  }

  .close-btn:hover {
    color: var(--text);
  }

  .modal-footer {
    padding: 14px 18px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    background: rgba(7, 10, 18, 0.5);
  }

  /* Progress Bars & Spinners */
  .progress-track {
    height: 4px;
    width: 100%;
    background: var(--bg-input);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--teal), var(--cyan));
    transition: width 0.2s ease-out;
  }

  .spinner {
    width: 15px;
    height: 15px;
    border: 2px solid rgba(255, 255, 255, 0.25);
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Scrollbars */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: transparent;
  }

  ::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: var(--border-light);
  }

  /*
   * Material Symbols icon, e.g. <span class="icon">search</span>. The font is
   * loaded in index.html; add new icon names to its icon_names list.
   */
  .icon {
    font-family: 'Material Symbols Rounded';
    font-weight: normal;
    font-style: normal;
    font-size: 1.25em;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    vertical-align: middle;
    user-select: none;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: 'liga';
  }
`;
