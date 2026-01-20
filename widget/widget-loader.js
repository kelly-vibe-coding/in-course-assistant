/**
 * Course Chat Widget - Floating Widget Loader
 *
 * This script is added once to your LMS (e.g., Skilljar Global Head Snippet).
 * It automatically detects if a course bot exists for the current course and
 * shows a floating chat widget if one is configured.
 *
 * Usage:
 * <script src="https://your-domain.com/widget-loader.js" data-api-url="https://your-domain.com"></script>
 */

(function() {
    'use strict';

    // Get API URL from script tag
    const scriptTag = document.currentScript;
    const apiUrl = scriptTag?.getAttribute('data-api-url') || '';

    if (!apiUrl) {
        console.warn('[InCourseAssistant] No data-api-url specified on script tag');
        return;
    }

    // Configuration (position will be updated from server config)
    const config = {
        apiUrl: apiUrl.replace(/\/$/, ''), // Remove trailing slash
        position: 'bottom-right', // Will be overridden by server config
        zIndex: 999999,
        bubbleSize: 72,
        widgetWidth: 380,
        widgetHeight: 550,
    };

    // State
    let isOpen = false;
    let courseData = null;
    let widgetContainer = null;
    let widgetIframe = null;
    let bubbleButton = null;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let initialX = 0;
    let initialY = 0;

    /**
     * Apply widget style settings from server config
     */
    function applyWidgetSettings(configData) {
        const style = configData?.config?.style || {};

        // Apply position setting
        if (style.position === 'bottom-left' || style.position === 'bottom-right') {
            config.position = style.position;
        }
    }

    /**
     * Detect LMS course ID from page
     * Supports: Skilljar (skilljarCourse.id)
     * Can be extended for other LMS platforms
     */
    function detectLmsCourseId() {
        // Skilljar
        if (typeof skilljarCourse !== 'undefined' && skilljarCourse?.id) {
            return { platform: 'skilljar', courseId: skilljarCourse.id };
        }

        // Add more LMS detection here as needed
        // Example: if (typeof someOtherLms !== 'undefined') { ... }

        return null;
    }

    /**
     * Check if a course bot exists for this LMS course
     */
    async function checkForCourseBot(lmsCourseId) {
        try {
            const response = await fetch(
                `${config.apiUrl}/api/widget/detect?lms_course_id=${encodeURIComponent(lmsCourseId)}`
            );

            if (response.ok) {
                return await response.json();
            }

            // 404 = no course bot configured, which is normal
            if (response.status === 404) {
                return null;
            }

            console.warn('[InCourseAssistant] Error checking for course bot:', response.status);
            return null;
        } catch (error) {
            console.warn('[InCourseAssistant] Failed to check for course bot:', error);
            return null;
        }
    }

    /**
     * Create the floating chat bubble button
     */
    function createBubble() {
        const colors = courseData?.config?.colors || {};
        const primaryColor = colors.primary || '#0D2B3E';
        const accentColor = colors.accent || '#5FCFB4';

        bubbleButton = document.createElement('button');
        bubbleButton.id = 'course-chat-bubble';
        bubbleButton.setAttribute('aria-label', 'Open course assistant');
        bubbleButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
        `;

        // Styles
        Object.assign(bubbleButton.style, {
            position: 'fixed',
            bottom: '20px',
            right: config.position === 'bottom-right' ? '20px' : 'auto',
            left: config.position === 'bottom-left' ? '20px' : 'auto',
            width: `${config.bubbleSize}px`,
            height: `${config.bubbleSize}px`,
            borderRadius: '50%',
            backgroundColor: primaryColor,
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.25)',
            zIndex: config.zIndex,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'transform 0.2s, box-shadow 0.2s',
            outline: 'none',
        });

        // Hover effects
        bubbleButton.addEventListener('mouseenter', () => {
            bubbleButton.style.transform = 'scale(1.1)';
            bubbleButton.style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.3)';
        });

        bubbleButton.addEventListener('mouseleave', () => {
            if (!isOpen) {
                bubbleButton.style.transform = 'scale(1)';
                bubbleButton.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.25)';
            }
        });

        bubbleButton.addEventListener('click', toggleWidget);

        document.body.appendChild(bubbleButton);
    }

    /**
     * Create the widget container with iframe
     */
    function createWidgetContainer() {
        widgetContainer = document.createElement('div');
        widgetContainer.id = 'course-chat-widget-container';

        // Current size (can be resized by user)
        let currentWidth = config.widgetWidth;
        let currentHeight = config.widgetHeight;

        // Store initial position for potential reset
        const initialBottom = config.bubbleSize + 30;
        const initialRight = config.position === 'bottom-right' ? 20 : null;
        const initialLeft = config.position === 'bottom-left' ? 20 : null;

        Object.assign(widgetContainer.style, {
            position: 'fixed',
            bottom: `${initialBottom}px`,
            right: initialRight !== null ? `${initialRight}px` : 'auto',
            left: initialLeft !== null ? `${initialLeft}px` : 'auto',
            width: `${currentWidth}px`,
            height: `${currentHeight}px`,
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.25)',
            zIndex: config.zIndex - 1,
            overflow: 'hidden',
            opacity: '0',
            transform: 'translateY(20px) scale(0.95)',
            transition: 'opacity 0.3s, transform 0.3s',
            pointerEvents: 'none',
            minWidth: '320px',
            minHeight: '400px',
            maxWidth: '90vw',
            maxHeight: '80vh',
        });

        // Create drag handle bar at the top
        const dragHandle = document.createElement('div');
        dragHandle.id = 'course-chat-drag-handle';
        Object.assign(dragHandle.style, {
            position: 'absolute',
            top: '0',
            left: '0',
            right: '0',
            height: '28px',
            background: 'linear-gradient(180deg, rgba(0,0,0,0.08) 0%, rgba(0,0,0,0) 100%)',
            cursor: 'move',
            zIndex: '15',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderTopLeftRadius: '12px',
            borderTopRightRadius: '12px',
        });

        // Add grip indicator dots
        const gripIndicator = document.createElement('div');
        Object.assign(gripIndicator.style, {
            width: '40px',
            height: '4px',
            background: 'rgba(0,0,0,0.2)',
            borderRadius: '2px',
            opacity: '0.6',
            transition: 'opacity 0.2s',
        });
        dragHandle.appendChild(gripIndicator);

        // Hover effect on grip
        dragHandle.addEventListener('mouseenter', () => {
            gripIndicator.style.opacity = '1';
        });
        dragHandle.addEventListener('mouseleave', () => {
            if (!isDragging) {
                gripIndicator.style.opacity = '0.6';
            }
        });

        // Drag functionality
        dragHandle.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragStartX = e.clientX;
            dragStartY = e.clientY;

            // Get current position - convert bottom/right to top/left for easier manipulation
            const rect = widgetContainer.getBoundingClientRect();
            initialX = rect.left;
            initialY = rect.top;

            // Switch to top/left positioning for dragging
            widgetContainer.style.bottom = 'auto';
            widgetContainer.style.right = 'auto';
            widgetContainer.style.left = `${initialX}px`;
            widgetContainer.style.top = `${initialY}px`;

            // Also switch bubble to top/left positioning
            const bubbleRect = bubbleButton.getBoundingClientRect();
            bubbleButton.style.bottom = 'auto';
            bubbleButton.style.right = 'auto';
            bubbleButton.style.left = `${bubbleRect.left}px`;
            bubbleButton.style.top = `${bubbleRect.top}px`;

            // Disable transition during drag
            widgetContainer.style.transition = 'none';
            bubbleButton.style.transition = 'none';

            // Prevent iframe from capturing mouse events
            widgetIframe.style.pointerEvents = 'none';

            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const deltaX = e.clientX - dragStartX;
            const deltaY = e.clientY - dragStartY;

            let newX = initialX + deltaX;
            let newY = initialY + deltaY;

            // Keep widget within viewport bounds
            const rect = widgetContainer.getBoundingClientRect();
            const maxX = window.innerWidth - rect.width;
            const maxY = window.innerHeight - rect.height;

            newX = Math.max(0, Math.min(newX, maxX));
            newY = Math.max(0, Math.min(newY, maxY));

            widgetContainer.style.left = `${newX}px`;
            widgetContainer.style.top = `${newY}px`;

            // Move bubble button to stay below the widget
            const bubbleX = newX + (rect.width / 2) - (config.bubbleSize / 2);
            const bubbleY = newY + rect.height + 10;
            bubbleButton.style.left = `${bubbleX}px`;
            bubbleButton.style.top = `${bubbleY}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                widgetContainer.style.transition = 'opacity 0.3s, transform 0.3s';
                bubbleButton.style.transition = 'transform 0.2s, box-shadow 0.2s';
                widgetIframe.style.pointerEvents = 'auto';
                gripIndicator.style.opacity = '0.6';
            }
        });

        // Create resize handle (top-left corner for bottom-right positioned widget)
        const resizeHandle = document.createElement('div');
        resizeHandle.id = 'course-chat-resize-handle';
        const isBottomRight = config.position === 'bottom-right';

        Object.assign(resizeHandle.style, {
            position: 'absolute',
            top: '0',
            left: isBottomRight ? '0' : 'auto',
            right: isBottomRight ? 'auto' : '0',
            width: '20px',
            height: '20px',
            cursor: isBottomRight ? 'nw-resize' : 'ne-resize',
            zIndex: '10',
            background: 'transparent',
        });

        // Add visual indicator on hover
        const resizeIndicator = document.createElement('div');
        Object.assign(resizeIndicator.style, {
            position: 'absolute',
            top: '4px',
            left: isBottomRight ? '4px' : 'auto',
            right: isBottomRight ? 'auto' : '4px',
            width: '12px',
            height: '12px',
            opacity: '0',
            transition: 'opacity 0.2s',
            borderTop: '2px solid rgba(0,0,0,0.3)',
            borderLeft: isBottomRight ? '2px solid rgba(0,0,0,0.3)' : 'none',
            borderRight: isBottomRight ? 'none' : '2px solid rgba(0,0,0,0.3)',
        });
        resizeHandle.appendChild(resizeIndicator);

        resizeHandle.addEventListener('mouseenter', () => {
            resizeIndicator.style.opacity = '1';
        });
        resizeHandle.addEventListener('mouseleave', () => {
            resizeIndicator.style.opacity = '0';
        });

        // Resize functionality
        let isResizing = false;
        let startX, startY, startWidth, startHeight;

        resizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            startWidth = widgetContainer.offsetWidth;
            startHeight = widgetContainer.offsetHeight;

            // Disable transition during resize
            widgetContainer.style.transition = 'none';

            // Prevent iframe from capturing mouse events
            widgetIframe.style.pointerEvents = 'none';

            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;

            let newWidth, newHeight;

            if (isBottomRight) {
                // For bottom-right: dragging top-left corner
                newWidth = startWidth - (e.clientX - startX);
                newHeight = startHeight - (e.clientY - startY);
            } else {
                // For bottom-left: dragging top-right corner
                newWidth = startWidth + (e.clientX - startX);
                newHeight = startHeight - (e.clientY - startY);
            }

            // Apply constraints
            newWidth = Math.max(320, Math.min(newWidth, window.innerWidth * 0.9));
            newHeight = Math.max(400, Math.min(newHeight, window.innerHeight * 0.8));

            widgetContainer.style.width = `${newWidth}px`;
            widgetContainer.style.height = `${newHeight}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                widgetContainer.style.transition = 'opacity 0.3s, transform 0.3s';
                widgetIframe.style.pointerEvents = 'auto';
            }
        });

        // Create iframe
        const widgetUrl = `${config.apiUrl}/widget.html?course_id=${courseData.course_id}&api_url=${encodeURIComponent(config.apiUrl)}`;

        widgetIframe = document.createElement('iframe');
        widgetIframe.src = widgetUrl;
        widgetIframe.setAttribute('title', 'Course Assistant');
        widgetIframe.setAttribute('allow', 'microphone');

        Object.assign(widgetIframe.style, {
            width: '100%',
            height: '100%',
            border: 'none',
        });

        widgetContainer.appendChild(dragHandle);
        widgetContainer.appendChild(resizeHandle);
        widgetContainer.appendChild(widgetIframe);
        document.body.appendChild(widgetContainer);
    }

    /**
     * Toggle widget open/closed
     */
    function toggleWidget() {
        isOpen = !isOpen;

        if (isOpen) {
            // Open
            widgetContainer.style.opacity = '1';
            widgetContainer.style.transform = 'translateY(0) scale(1)';
            widgetContainer.style.pointerEvents = 'auto';

            // Change bubble to close icon
            bubbleButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            `;
            bubbleButton.setAttribute('aria-label', 'Close course assistant');
        } else {
            // Close
            widgetContainer.style.opacity = '0';
            widgetContainer.style.transform = 'translateY(20px) scale(0.95)';
            widgetContainer.style.pointerEvents = 'none';

            // Change back to chat icon
            bubbleButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            `;
            bubbleButton.setAttribute('aria-label', 'Open course assistant');
        }
    }

    /**
     * Handle clicks outside the widget to close it
     */
    function handleClickOutside(event) {
        if (!isOpen) return;

        const clickedBubble = bubbleButton?.contains(event.target);
        const clickedWidget = widgetContainer?.contains(event.target);

        if (!clickedBubble && !clickedWidget) {
            toggleWidget();
        }
    }

    /**
     * Handle escape key to close widget
     */
    function handleEscapeKey(event) {
        if (event.key === 'Escape' && isOpen) {
            toggleWidget();
        }
    }

    /**
     * Initialize the widget
     */
    async function init() {
        // Detect LMS course
        const lmsInfo = detectLmsCourseId();

        if (!lmsInfo) {
            // No LMS course detected - this is normal on non-course pages
            return;
        }

        // Check if a course bot exists for this course
        courseData = await checkForCourseBot(lmsInfo.courseId);

        if (!courseData) {
            // No course bot configured for this course - do nothing
            return;
        }

        // Apply settings from server config (e.g., position)
        applyWidgetSettings(courseData);

        // Course bot found - create the floating widget
        createBubble();
        createWidgetContainer();

        // Event listeners
        document.addEventListener('click', handleClickOutside);
        document.addEventListener('keydown', handleEscapeKey);

        console.log('[InCourseAssistant] Widget loaded for course:', courseData.course_name);
    }

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // Small delay to ensure LMS variables are set
        setTimeout(init, 100);
    }
})();
